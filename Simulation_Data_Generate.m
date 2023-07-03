function [B,s_real,Result] = Simulation_Data_Generate (LFM,Cortex,Time,OPTIONS)
% Descriptions: Genarate simulated EEG/MEG data for extended sources
% Inputs: LFM: Lead field Matrix(#sensors X #sources)
%         Cortex.| Vertices
%               .| Faces    Cortex files
%         Time: Time for each sample
%         OPTIONS. DefinedArea: Areas for each patch
%                . seedvox    : seedvoxs of each patch
%                . frequency  : frequency of the gaussian damped time courses
%                . tau        : time delay of the gaussian damped time courses
%                . omega      : variation of the gaussian damped time courses
%                . Amp        : Amplitude of the gaussian damped time courses
%                . uniform    : applying uniform activations (uniform = 1) or not (uniform = 0)
%                . WGN        : adding white gaussian noise (WGN = 1) or not (WGN = 0)

% Version 1: Liu Ke, 2018/8/22
%% ===== DEFINE DEFAULT OPTIONS =====
Def_OPTIONS.WGN         = 1;
Def_OPTIONS.uniform     = 1;
Def_OPTIONS.ar          = 0;
Def_OPTIONS.GridLoc     = [];
Def_OPTIONS.MixMatrix = eye(4);%eye(numel(tau));
% Copy default options to OPTIONS structure (do not replace defined values)
OPTIONS = struct_copy_fields(OPTIONS, Def_OPTIONS, 0);

GridLoc = OPTIONS.GridLoc;
AreaDef = OPTIONS.DefinedArea;
seedvox = OPTIONS.seedvox;
f = OPTIONS.frequency;
tau = OPTIONS.tau;
omega = OPTIONS.omega;
A = OPTIONS.Amp;
Uniform = OPTIONS.uniform;
WGN = OPTIONS.WGN;
SNR = OPTIONS.SNR;
ar = OPTIONS.ar;
MixMatrix = OPTIONS.MixMatrix;
SinglePoint = OPTIONS.SinglePoint;
nSource = size(LFM,2);
%% Active Vox    
ActiveVoxSeed = num2cell(seedvox);
ActiveVox = [];
[~, VertArea] = tess_area(Cortex.Vertices, Cortex.Faces);
Cortex.VertConn = tess_vertconn(Cortex.Vertices, Cortex.Faces);
for k = 1:numel(seedvox)
    ActiveVoxSeed{k} = PatchGenerate(seedvox(k),Cortex.VertConn,VertArea,AreaDef(k));
    ActiveVox = union(ActiveVoxSeed{k},ActiveVox);
end
Area = sum(VertArea(ActiveVox));
%% ------------------ Simulation data ---------------------%
if ~SinglePoint
    StimTime = find(abs(Time) == min(abs(Time)));
    s_real = zeros(nSource,numel(Time));
    Activetime = StimTime+1:numel(Time);
    % -----------Gaussian Damped sinusoidal time courses------------------%
    if ~ar
        Basis = zeros(numel(tau),numel(Time));
        for k = 1:numel(tau)
            Basis(k,Activetime) = sin(2*pi*f(k)*(Time(Activetime))).*exp(-((Time(Activetime)-tau(k))/omega(k)).^2);
        end
        % Basis = Basis./repmat(max(Basis')',1,size(Basis,2));
        Basis = orth(Basis')';
        %Basis = Basis./repmat(sqrt(sum(Basis.^2,2)),1,size(Basis,2));
    %     Basis = Basis./repmat(max(Basis')',1,size(Basis,2));
        AA = MixMatrix*A;
        % % ========================Uniform/NonUniform Sources ==============================%
        if Uniform
            for k = 1:numel(seedvox)
                s_real(ActiveVoxSeed{k},:) = repmat(AA(k,:)*Basis,numel(ActiveVoxSeed{k}),1);
            end
        else
            for k = 1:numel(seedvox)
                %                s_real(ActiveVoxSeed{k},:) = 1e6*VertArea(ActiveVoxSeed{k})*AA(k,:)*Basis;
                dis = sqrt(sum((GridLoc(ActiveVoxSeed{k},:)-repmat(GridLoc(seedvox(k),:),numel(ActiveVoxSeed{k}),1)).^2,2));
                Amplitude = exp(-dis.^2./(0.6*max(dis)).^2);
                s_real(ActiveVoxSeed{k},:) = Amplitude*AA(k,:)*Basis;
            end
        end
        % % ======================================================================%

        % %===================== Current Density Model (CDM)======================%
        %     for i =1 :numel(ActiveVox)
        %s_real(ActiveVox(i),Activetime) = 1e6*VertArea(ActiveVox(i))*1e-10*sin(2*pi*f*Time(Activetime));
        %     end
    %% AR time series    
    else
        cfg             = [];
        cfg.ntrials     = 1;
        cfg.triallength = max(Time);
        cfg.fsample     = 1/(Time(2) - Time(1));
        cfg.nsignal     = size(OPTIONS.noisecov,1);
        cfg.method      = 'ar';

        cfg.params      = OPTIONS.params;
        cfg.noisecov    = OPTIONS.noisecov;

        artimeseries              = ft_connectivitysimulation(cfg);
        s_active = zeros(numel(seedvox),numel(Time));
        s_active(:,Activetime) = artimeseries.trial{1}(1:numel(seedvox),:);
    %     for i = 1:size(s_active,1)
    %     s_active(i,Activetime) = ft_preproc_bandpassfilter(artimeseries(i,:), size(artimeseries(i,:),2), [1 30]);
    %     end
        for k = 1:numel(seedvox)
            s_real(ActiveVoxSeed{k},:) = A*repmat(s_active(k,:),numel(ActiveVoxSeed{k}),1);
        end
    end
else
    s_real = zeros(nSource,1);
    A = 8*1e-9;
    for k = 1 : numel(seedvox)
        s_real(ActiveVoxSeed{k},:) = A;
    end
end
% % ======================================================================%

% ===================== Realistic EEG background noise ===================%
%load 'HumanEEGNoise.mat'



% % ======================================================================%
% ===========================add pinknoise=============================== %
% flag = 1;
% while flag
%     ind = randperm(nSource); 
%     %5个大小为3~5cm^2面积的区域，作为5个cluster的面积�?
%     %注意：spm中计算的Cortex.VertArea是以毫米为单位的,1cm^2= 100mm^2,1个vert�?��的面积大概为20mm^2.
%     pink_nclus =5;
%     pink_Area = unifrnd(3,5,1,pink_nclus)*1e-4;
%     %在Vertcies中随机�?5个作为种子点
%     pink_seedvox = ind(1:pink_nclus);
%     pink_ActiveVoxSeed = num2cell(pink_seedvox);
%     pink_ActiveVox = [];
%      for k = 1:numel(pink_seedvox)
%             pink_ActiveVoxSeed{k} = PatchGenerate(pink_seedvox(k),Cortex.VertConn,VertArea,pink_Area(k));%在seedvod处生成大小为DefinedArea的patch
%             pink_ActiveVox = union(pink_ActiveVoxSeed{k},pink_ActiveVox);%�?个patch合并到一个列表，�?81个网格�?
%      end
%      flag = numel(intersect(pink_ActiveVox,ActiveVox));
% end
% %pn为活跃vertices的时间过�?
% pn = mkpinknoise(1,numel(pink_ActiveVox))';  % 
% %转换脑源噪声,并且归一化处�?
% EEG_brain_noise = zeros(nSource,1); % 
% EEG_brain_noise(pink_ActiveVox,:) = pn;
% %EEG_brain_noise = EEG_brain_noise ./ norm(EEG_brain_noise,'fro');
% %==================================脑源信号+脑源噪声，映射到头皮，转换为传感器信�?============================
% SNIR = 0;
% ri = 1/10^(SNIR/10);    % SNIR = 10lg(Ps/Pn)
% EEG_brain_signal = s_real + ri*EEG_brain_noise;
% EEG_sensor_signal = LFM*EEG_brain_signal;

if  ~WGN
    load 'ArtHumanEEGNoise.mat'
    noise = noise(:,1:size(s_real,2));
    r = norm(LFM*s_real,'fro')/norm(noise,'fro')/10^(SNR/20);
    Noise = r*noise;
    B = LFM*s_real + Noise;
else
    % %==================== White Gaussian Noise ============================= %
    B = awgn(LFM*s_real,SNR,'measured');
end
% =======================================================================%
Result.B = B;
Result.real = s_real;
Result.seedvox = seedvox;
Result.ActiveVox = ActiveVox;
Result.ActiveVoxSeed = ActiveVoxSeed;
Result.Area = Area;
    

function noise= mkpinknoise(n,m)
% makes m channels of pink noise of length n. Each column is one channel
%
% Guido Nolte, 2012-2015
% g.nolte@uke.de

% If you use this code for a publication, please ask Guido Nolte for the
% correct reference to cite.

% License
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see http://www.gnu.org/licenses/.

% randn('state',sum(100*clock))
n1=2*ceil(n/2)+1;
scal=sqrt(1./[1:(n1-3)/2]');
ff=zeros(n1,m);
ff(2:(n1-1)/2,:)=repmat(scal,1,m);
noise=fft(randn(n1,m));
noise=2*real(ifft(noise.*ff));
noise=noise(1:n,:);