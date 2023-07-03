function [S] = Lp_ADMM(B,L,VertConn,sparseweight,varargin)
%% Description: Reconstructed extended sources based on Lp-norm regularization
% Data Model:
% B = LS + epsilon;
% U = V*S; variation sources

% Input:
%         B(d_b x 1):               M/EEG Measurement
%         L(d_b x d_s):             Leadfiled Matrix
%         VertConn:                 Cortex Connectivity Condition
%         sparseweight:             sparseweight for the sparse variation
%                                   (typically = 0.01)

% Output:
%         S:                        Estimated Sources
%%
[nSensor,nSource] = size(L);
nSnap = size(B,2);
tol = 1e-3;
QUIET = 1;
rou_update = 1;
rou = 1e17; 
p = 1;
% sparseweight = 0.05;
% get input argument values
if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'transform'
                transform = varargin{i+1};
            case 'tol'
                tol = varargin{i+1};
            case 'roupar'
                rou = varargin{i+1};
            case 'p'
                p = varargin{i+1};
            case 'lam'
                lam = varargin{i+1};
            case 'methods'
                flag = varargin{i+1};
        end
    end
end       
  Edge = VariationEdge(VertConn);      
if strcmp(transform, 'Variation')
    M = VariationEdge(VertConn);
elseif strcmp(transform, 'Laplacian')
    NVertConn = sum(VertConn,2);
    M = bsxfun(@minus,spdiags(ones(nSource,1),0,nSource,nSource),bsxfun(@times,bsxfun(@rdivide,VertConn,NVertConn),0.95*ones(nSource,1)));
elseif strcmp(transform, 'Laplacian+Variation')
    NVertConn = sum(VertConn,2);
    M = bsxfun(@minus,spdiags(ones(nSource,1),0,nSource,nSource),bsxfun(@times,bsxfun(@rdivide,VertConn,NVertConn),0.95*ones(nSource,1)));
    M = [opts.laplacian*M;VariationEdge(VertConn)];
elseif strcmp(transform,'Sparse+Laplacian')
    NVertConn = sum(VertConn,2);
    M = bsxfun(@minus,spdiags(ones(nSource,1),0,nSource,nSource),bsxfun(@times,bsxfun(@rdivide,VertConn,NVertConn),0.95*ones(nSource,1)));
    M = [sparseweight*sparse(1:nSource,1:nSource,1);M];
elseif strcmp(transform,'Sparse+Variation')
    M = [sparseweight*sparse(1:nSource,1:nSource,1);VariationEdge(VertConn)];
elseif strcmp(transform, 'Sparse')
    M = sparse(1:nSource,1:nSource,1);
end

%%

ADMM_iter = 600;

% Initial values
TMNE = MNE(B,[],L,[],'MNE');
S_MNE = TMNE*B;
S = S_MNE + 0.02*max(S_MNE)*randn(size(L,2),1);
% S = 0.001*randn(size(L,2),1);
% S = 0.0001*randsrc(size(L,2),1);
U = M*S;  U_old = U;
Z = zeros(size(M,1),nSnap);

rou_old = rou;

Lambda_MAX = eigs(M'*M,1,'lm');%norm(full(M),2).^2;
LLt =L*L';LtB = L'*B;
mu = 0.9/(rou/2*Lambda_MAX);%0.9/rou/NormM; 
Precision = 2*(mu*speye(nSource) - mu^2*L'/(eye(nSensor) + mu*LLt)*L);
tic
% for iter = 1 : MAX_iter
%% ADMM (Source Update)
alpha = 0.6; 
    for iter_ADMM = 1:ADMM_iter
%---------------- S update ---------------------------------%           
        S = Precision*(2*LtB + (1/mu)*S - rou*M'*(M*S - U + Z));
%---------------- U update ---------------------------------%     
        MS = M*S;
        
%        lam = 10*rou;
        MS_hat = alpha*MS + (1- alpha)*U_old;
        U = prox(MS_hat + Z,p,lam,rou,flag);        

%---------------- Z update ---------------------------------%     
        Z = Z + (MS_hat - U);         
%---------------- stop creterion  ---------------------------------%        
        primerror = norm(MS - U,'fro');
        dualerror = norm(rou*M'*(U - U_old),'fro');
        U_old = U;
        
        tol_prim = 1e-6*max(norm(U,'fro'),norm(MS,'fro'));
        tol_dual = 1e-6*rou*norm(M'*Z,'fro');
 
        Rprimerror = primerror/max(norm(U,'fro'),norm(MS,'fro'));
        Rdualerror = dualerror/norm(rou*M'*Z,'fro');
        
        if primerror < tol_prim && dualerror < tol_dual
            break;
        end        
%---------------- rou update ---------------------------------%   
        if rou_update && mod(iter_ADMM,10) == 0
            ratio = -1;
            if Rdualerror~=0
                ratio = sqrt(Rprimerror/Rdualerror);
            end
            
             tau_max  = 2;
            if ratio>=1 && ratio < tau_max, tau = ratio;
            elseif ratio> 1/tau_max && ratio < 1, tau = 1/ratio;
            else tau = tau_max;
            end
%                tau  = 2;
            if Rprimerror > 10 *  Rdualerror;
                rou = tau*rou; Z = Z./tau;
            elseif Rdualerror > 10 * Rprimerror
                rou = rou/tau; Z = Z.*tau;
            end
            if ~QUIET
                fprintf('rou = %g, Rprimmerror = %g, Rdualerror = %g\n',rou,Rprimerror,Rdualerror);
            end
            if rou ~= rou_old
                mu = 0.9/(rou*Lambda_MAX);%0.9/rou/NormM;
                Precision = mu*speye(nSource) - mu^2*L'/(eye(nSensor) + mu*LLt)*L;
            end
            rou_old = rou;
        end
        
        if ~mod(iter_ADMM,100)
            SS{iter_ADMM/100} = S;
        end
    end 

% fprintf('ADMM-Lp : iteration =  %g, MSE =  %g,  lam =  %g\n',norm(S - Sold,'fro')/norm(S,'fro'),lam);
% if norm(S - Sold,'fro')/norm(S,'fro') < tol, break; end

% Sold = S;
% SS{iter} = S;
 toc   
% end

S = SS;



function  M = VariationEdge(VertConn)
nSource = size(VertConn,1);
nEdge = numel(find(VertConn(:)~=0))/2;
M = sparse(nEdge,nSource);
edge = 0;
for i = 1:nSource
    idx = find(VertConn(i,:)~=0);
    idx = idx(idx<i);
    for j = 1:numel(idx)
        M(j+edge,i) = 1;
        M(j+edge,idx(j)) = -1;     
    end
    edge = edge + numel(idx);
end

function Z = prox(Y,p,lam,rou,flag)
if p == 1
    thr = lam/rou;
    Z = abs(Y);
    Z(Z<=thr) = 0;
    ind = find(Z>0);
    Z(ind) = abs(Y(ind)) - thr;
    Z = sign(Y).*Z;
elseif p < 1 && p > 0
    k = 3;
    temp = lam/rou;
    thr = (2*temp*(1-p))^(1/(2-p)) + temp*p*(2*temp*(1-p))^((p-1)/(2-p))
    Z = abs(Y);
    Z(Z<=thr) = 0;
    ind = find(Z>0);
    for i = 1:k
        Z(ind) = abs(Y(ind)) - temp*p*Z(ind).^(p-1);
    end
    Z = sign(Y).*Z;
end
    
% if strcmp(flag, 'Lp')
% %% Proximal for Lp-norm
% % Ref: Zuo Wangmeng et.al. A Generalized Iterated Shrinkage Algorithm for
% % Non-convex Sparse Coding, 2013.
%     k = 2;
%     temp = lam/rou;
%     thr = (2*temp*(1-p))^(1/(2-p)) + temp*p*(2*temp*(1-p))^((p-1)/(2-p))
%     Z = abs(Y);
%     Z(Z<=thr) = 0;
%     ind = find(Z>0);
%     for i = 1:k
%         Z(ind) = abs(Y(ind)) - temp*p*Z(ind).^(p-1);
%     end
%     Z = sign(Y).*Z;
% 
% %% IRLS L2
% %     iter = 50;
% %     Z = Y;
% %     Z_old = Y;
% %     for i = 1:iter
% %         %     w = (lam/rou)*(Z.^2+1e-20).^(p/2-1);
% %         w = (lam/rou)*(Z.^2+1e-6*max(Z.^2)).^(p/2-1);
% %         Z = diag((1+w).^(-1))*Y;
% %         MSE = norm((Z-Z_old),'fro')/norm(Z,'fro');
% %         if MSE<1e-3
% %             break;
% %         end
% %         Z_old = Z;
% %     end
% %     fprintf('MSE = %g, iter = %g\n',MSE,iter);
% 
% elseif strcmp(flag, 'L1')
% %% Soft thresholding for L1-norm (LASSO)
%     thr = lam/rou;
%     Z = Y;
%     ind1 = find(Z < -thr);
%     ind2 = find(abs(Z) < thr);
%     ind3 = find(Z > thr);
%     Z(ind1) = Y(ind1) + thr;
%     Z(ind2) = 0;
%     Z(ind3) = Y(ind3) - thr;

% end

