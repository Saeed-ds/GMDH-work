clc;
clear;
close all;

%% ===================== LOAD DATA =====================
T = readtable('machine.data', ...
    'FileType','text', ...
    'Delimiter',',', ...
    'ReadVariableNames', false);

T.Properties.VariableNames = ...
 {'Vendor','Model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP'};

% Unit normalization
Tn = T;
Tn.MYCT = T.MYCT / 10;
Tn.MMIN = T.MMIN / 1024;
Tn.MMAX = T.MMAX / 1024;
Tn.CACH = T.CACH / 1024;
Tn.PRP  = T.PRP  / 10;

X = Tn{:,{'MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX'}}';
Y = Tn.PRP';

%% ===================== TRAIN / VALIDATION SPLIT =====================
trainRatio = 0.7;
numSamples = size(X,2);
numTrain = round(trainRatio * numSamples);

Xtrain = X(:,1:numTrain);
Ytrain = Y(1:numTrain);

Xval = X(:,numTrain+1:end);
Yval = Y(numTrain+1:end);

%% ===================== USER INPUT =====================
Lmax = input('Enter maximum number of layers: ');
MaxNeurons = input('Enter max neurons per layer: ');

%% ===================== INITIALIZATION =====================
Xlayer = Xtrain;
Xval_layer = Xval;

% ---- manual normalization ----
mu = mean(Xlayer,2);
sigma = std(Xlayer,0,2);
sigma(sigma==0) = 1;
Xlayer = (Xlayer - mu) ./ sigma;

mu = mean(Xval_layer,2);
sigma = std(Xval_layer,0,2);
sigma(sigma==0) = 1;
Xval_layer = (Xval_layer - mu) ./ sigma;

bestMSE = inf;
patience = 2;
badLayers = 0;

valMSE = zeros(1,Lmax);
Yval_pred_layers = cell(Lmax,1);

%% ===================== GMDH LAYERS =====================
for layer = 1:Lmax

    numFeatures = size(Xlayer,1);
    numTrainSamples = size(Xlayer,2);
    numValSamples = size(Xval_layer,2);
    numNeurons = nchoosek(numFeatures,2);

    if numNeurons == 0
        break;
    end

    Yp_train = zeros(numTrainSamples,numNeurons);
    Yp_val   = zeros(numValSamples,numNeurons);

    r = 1;
    for i = 1:numFeatures
        for j = i+1:numFeatures

            % -------- TRAIN --------
            Xn = [ones(numTrainSamples,1), ...
                  Xlayer(i,:).', ...
                  Xlayer(j,:).', ...
                  (Xlayer(i,:).*Xlayer(j,:)).', ...
                  (Xlayer(i,:).^2).', ...
                  (Xlayer(j,:).^2).'];

            a = pinv(Xn) * Ytrain.';
            Yp_train(:,r) = Xn * a;

            % -------- VALIDATION --------
            Xv = [ones(numValSamples,1), ...
                  Xval_layer(i,:).', ...
                  Xval_layer(j,:).', ...
                  (Xval_layer(i,:).*Xval_layer(j,:)).', ...
                  (Xval_layer(i,:).^2).', ...
                  (Xval_layer(j,:).^2).'];

            Yp_val(:,r) = Xv * a;

            r = r + 1;
        end
    end

    %% ===================== NEURON SELECTION =====================
    valRMSE = sqrt(mean((Yp_val - Yval.').^2,1));
    [sortedRMSE, idx] = sort(valRMSE);

    numKeep = min(MaxNeurons, length(idx));
    keepIdx = idx(1:numKeep);

    %% ===================== UPDATE LAYERS =====================
    Xlayer = Yp_train(:,keepIdx).';
    Xval_layer = Yp_val(:,keepIdx).';

    % ---- normalize both ----
    mu = mean(Xlayer,2);
    sigma = std(Xlayer,0,2);
    sigma(sigma==0) = 1;
    Xlayer = (Xlayer - mu) ./ sigma;

    mu = mean(Xval_layer,2);
    sigma = std(Xval_layer,0,2);
    sigma(sigma==0) = 1;
    Xval_layer = (Xval_layer - mu) ./ sigma;

    Yval_pred_layers{layer} = Yp_val(:,keepIdx);
    valMSE(layer) = mean(sortedRMSE(1:numKeep).^2);

    fprintf('Layer %d | Validation MSE = %.6f\n', layer, valMSE(layer));

    %% ===================== EARLY STOPPING =====================
    if valMSE(layer) < bestMSE
        bestMSE = valMSE(layer);
        badLayers = 0;
    else
        badLayers = badLayers + 1;
        if badLayers >= patience
            fprintf('Early stopping at layer %d\n', layer);
            break;
        end
    end
end

%% ===================== PLOT =====================
[~,idxSort] = sort(Yval);
Ytrue_sorted = Yval(idxSort);

figure; hold on;
colors = jet(layer);

for l = 1:layer
    Ypred = mean(Yval_pred_layers{l},2);
    plot(Ypred(idxSort),'Color',colors(l,:),'LineWidth',1.5);
end

plot(Ytrue_sorted,'k--','LineWidth',2);
grid on;
xlabel('Validation Sample (sorted)');
ylabel('Output');
title('GMDH Validation Prediction per Layer');
legend([arrayfun(@(x) sprintf('Layer %d',x),1:layer,'UniformOutput',false),{'True'}]);