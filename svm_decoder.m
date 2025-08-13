%% run SVM decoder between two behaviors 
% Inputs:
%   neural population activity (cell x time)
%   behavioral matrix (behavior x time) 
%   bvlist (which two behaviors to decode)
%   window (time window for each behavioral bout to take average over)
%   fr (sample rate)
%   nshuf (number of times to shuffle)
% Example usage:
%   [acc,shuf,auc,aucshuf,nbout] = svm_decoder(ca,bv,[1,2],[-0.1,0.1],15,100)
% Outputs:
%    acc - accuracy of prediction 
%    shuf - average accuracy in shuffle control
%    auc - auROC curve for prediction
%    aucshuf - average auROC curve in shuffle control
%    nbout - number of bouts (minimum of all classes)


function [acc,shuf,auc,aucshuf,nbout] = svm_decoder(ca,bv,bvlist,window,fr,nshuf)
acc = []; shuf = []; nbout = [];
shift = randi([round(15*fr),size(ca,1)-round(15*fr)],1,nshuf);
[acc,nbout,auc] = prediction_bout(ca,bv,bvlist,window,fr);
parfor i = 1:length(shift)
    shifted = circshift(ca,shift(i),1); 
    [shuf(i),~,aucshuf(i)] = prediction_bout(shifted,bv,bvlist,window,fr);
end
shuf = mean(shuf);
aucshuf = mean(aucshuf);
f = plot_accuracy_bout(auc,aucshuf);
end
function [acrc,nbout,auc] = prediction_bout(ca,bv,bvlist,window,fr)
% make bouts
X = []; Y = []; % x = neuron x trial
onsets_all = [];
for i = 1:length(bvlist)
    CC = bwconncomp(bv(:,bvlist(i)));
    for b = 1:length(CC.PixelIdxList)
        onset= CC.PixelIdxList{b}(1);
        if onset + round(window(1)*fr) <= 0 || onset+round(window(2)*fr) > length(ca)
            continue % out of bound 
        end
        onsets_all = [onsets_all,onset];
        X = cat(2,X,mean(ca(onset+round(window(1)*fr):onset+round(window(2)*fr),:),1)');
        Y = [Y,i];
    end
end
n_b = [sum(Y==1),sum(Y==2)];
nbout = min(n_b);
nt = size(X,2);
yhat = zeros(nt,1);
score = zeros(nt,2);
for k = 1:nt % leave one out cross validation
    test_idx = k;
    % exclude the bouts within 15 seconds
    excl = abs(onsets_all(k) - onsets_all) < 15*fr;
    train_idx = ~excl;
    %disp(sum(train_idx))
    Y_tr = Y(train_idx);X_tr = X(:,train_idx);
    % balance training
    [X_tr,Y_tr,nbout] = balance_xy(X_tr,Y_tr);
    SVMModel = fitcsvm(X_tr',Y_tr');
    [yhat(test_idx),score(test_idx,:)] = predict(SVMModel,X(:,test_idx)');
end
acrc = mean(yhat==Y');
[~,~,~,auc] = perfcurve(Y,score(:,1),1);
end
function [X,Y,nbout] = balance_xy(X,Y)
nclass = length(unique(Y));
n_b = sum(Y==[1:nclass]',2);
nbout = min(n_b);
sampled = [];
for c = 1:nclass
    idx = find(Y == c);
    samp = randperm(sum(Y == c), nbout);
    sampled = [sampled,idx(samp)];
end
Y = Y(sampled); X = X(:,sampled);
end

function f = plot_accuracy_bout(acrc,shuf)
f = figure;
col = lines(2);
plot([acrc;shuf]'); xticks(1:2); xticklabels({'data','shuffle'}); xlim([0.5 2.5]); box off; hold on
ylabel('auROC');
end