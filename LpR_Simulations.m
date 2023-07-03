clc
clear
tic
%brainstorm %norgui
%%
Uniform = 1; % Uniform/NonUniform Sources

load Cortex.mat
load GridLoc.mat
load Gain.mat

Atlas = Cortex.Atlas(2);
[nSensor,nSource] = size(Gain);

VariousSNRs = 1;
VariousChannels = 0;

%% Make output directory structure for imaging results (if doesn't already exist)
if VariousSNRs
   scenario = 'various SNRs';%'Experimental Noise';%
   SNR1 = [-5, 0, 5, 10];
   p_values = 0.6*ones(4,1);
   condition = SNR1';
   K = 1*ones(4,1);
   DefinedArea = 6*1e-4*ones(size(condition,1),max(K));
elseif VariousChannels
   scenario = 'various channels';
   SNR1 = 5*ones(4,1);
   p_values = 0.6*ones(4,1);
   condition = [62, 47, 31, 16]';
   K = ones(4,1);
   DefinedArea = 6*1e-4*ones(size(condition,1),max(K));
end

%% Iteration
    dim = 0;
    Miter = 50 - dim;   
    Eccentricity = sqrt(sum(GridLoc.^2,2));
    Ec = find(Eccentricity > 70*1e-3);
    EstimatedArea_Mean = zeros(Miter,5);
for iter = 1:Miter    
    ind = randperm(numel(Ec));
for iteration = 1:size(condition,1)
    fprintf('iter = %g, iteration = %g\n', iter,iteration)
%     SD = [];DLE = []; PRE = []; RMSE = []; AUC = [];

%% Generate Simulated EEG Data
SNR = SNR1(iteration);
p = p_values(iteration);
seedvox = Ec(ind(1:K(iteration)));
tau = [0.1 0.35 0.5 0.6];omega = [0.1 0.15 0.15 0.15];
f = [10 11 8 9];
Amp = 1e-8;

TimeLen = 100;
Time = [-0.98:1/50:1];
StimTime = find(abs(Time) == min(abs(Time)));


OPTIONS.DefinedArea    = DefinedArea(iteration,:);
OPTIONS.seedvox        = seedvox;
OPTIONS.frequency      = f;
OPTIONS.tau            = tau;
OPTIONS.omega          = omega;
OPTIONS.Amp            = Amp;
OPTIONS.GridLoc        = GridLoc;
OPTIONS.uniform       = Uniform;
OPTIONS.WGN           = 1;
OPTIONS.SNR           = SNR;
OPTIONS.ar            = 0;
OPTIONS.params(:,:,1) = [ 0.8    0    0 ;
                            0  0.9  0.5 ;
                          0.4    0  0.5];

OPTIONS.params(:,:,2) = [-0.5    0    0 ;
                            0 -0.8    0 ;
                            0    0 -0.2];

OPTIONS.noisecov      = [ 0.3    0    0 ;
                            0    1    0 ;
                            0    0  0.2];
OPTIONS.SinglePoint   = 0;


[Data,s_real,Result] = Simulation_Data_Generate(Gain,Cortex,Time,OPTIONS);
ActiveVoxSeed = Result.ActiveVoxSeed;

% %=======================================================================%
fprintf('Actual SNR is %g\n',20*log10(norm(Gain*s_real,'fro')/norm(Data-Gain*s_real,'fro')));

%% Data scale
Scale = 1;
if Scale
    ScaleType = 0;  % ScaleType = 1, Simultaneously scale the MEG data and leadfiled matrix; ScaleType = 0, only scale the MEG data
    ratio = 2e-8;%Ratio(iiter); 6e-8
    if ScaleType == 0
        B = Data./ratio;
        Gain_scale = Gain;
    else
        B  = Data./ratio;
        Gain_scale = Gain./ratio;
        ratio = 1;
    end
else
    ratio = 1;
    B = Data;
    Gain_scale = Gain;
end
L = Gain_scale;
%% Source Estimation
MetricsInterval = [];

se = max(s_real);
[~,pos] = max(se);

if VariousChannels
    if ~any(ResultsLoading)
        acnumber = size(B,1);
        cnumber = condition(iteration);
        channels = sort(randsample(acnumber,cnumber));
        B = B(channels,:);
        L = L(channels,:);
        Result.channels = channels;
    else
        channels = Result.channels;
        B = B(channels,:);
        L = L(channels,:);
    end
end

%% VSSI-LpR ADMM solver
Weight = logspace(-4,0,20);
% Sourcenorm = zeros(10,1);variationnorm = zeros(10,1);
variation = 'Variation';%'Sparse+Variation';%'Sparse';% 'Laplacian';%'Laplacian+Variation';%'Sparse+Laplacian';%'Laplacian';%
opts.sparse = 0.25;%Weight(14);%Weight(8);%0.05;%%0.15;
% opts.laplacian = 0.8;

methods = 'Lp';
lam_lp = 4e4;

[S_LpR] = Lp_ADMM(B(:,pos),L,Cortex.VertConn,opts.sparse,'transform',variation,'tol',1e-4,'roupar',1e17,'lam',lam_lp/2,'p',p,'methods',methods);
S_vssilpr = S_LpR{end};
S_vssilpr = S_vssilpr*ratio;

[SD(1),DLE(1),nRMSE(1)]...
    = PerformanceMetric(GridLoc,S_vssilpr,s_real(:,pos),ActiveVoxSeed);%,'interval',MetricsInterval);
Roc = ROCextent(s_real(:,pos),S_vssilpr,Cortex,seedvox);
AUC(1) = median(Roc.mean);
EV(1) =  1 - norm(B(:,pos)*ratio - L*S_vssilpr,'fro')^2/norm(B(:,pos)*ratio,'fro')^2;
Result.VSSILPR = S_vssilpr;

end
end
