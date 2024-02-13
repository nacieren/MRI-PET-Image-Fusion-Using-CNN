clc;
clear;
close all;
%https://www.mathworks.com/help/images/create-image-datastore-containing-dicom-images.html

%dataFolder = '/MATLAB Drive/HW/datasetnii_MRI';
dataFolder = '/MATLAB Drive/HW/datasetnii_CT';



imds = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource' ...
    , 'foldernames',...
    'FileExtensions', '.nii',...
    'ReadFcn', @(x)script6_niftiReadCustomized(x));


% verisetindeki imageların boyutuna baktık
I = readimage(imds, 5);
size(I)




figure

for i = 1:2
    A = niftiread(imds.Files{i});
    subplot(1,2,i);
    imagesc(A(:,:,45));
    drawnow;
end




numTrainingFiles = 3;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');
%crossvalidation .....



layers = [ ...
    imageInputLayer([79 95 1])
    % derinlik sayısı bu örnekte 1
    convolution2dLayer(5,20)  % 5 filter size, 20 number of filters
    reluLayer                 % ReLU katmanı ile bir önceki convoltuion sonucunda x<0 olan degerler 0 a eşitlenir. 
    maxPooling2dLayer(2,'Stride',2)%  creates a max pooling layer with pool size [2 2] and stride [2 2]
                                   %  bu işlem sonucunda 14x14 lük 20 adet image a dönüşür 

    % conv-relu-pool tekrarlayarak derinlik sayısını arttırabilirsiniz
    %convolution2dLayer(5,20) 
    %reluLayer 
    %maxPooling2dLayer(2,'Stride',2) 
    % 


    fullyConnectedLayer(2)     % verisetindeki sınıf sayısı 10 olduğu için 10 olacak.
                                % tumor var-yok  buradaki sayı 2 olacak 
    softmaxLayer                %  son katmanın değerini 0-1 arasına normalize etmek için kullanılan fonksiyon
    classificationLayer];      % computes the cross-entropy loss for classification .. threshold belirleyerek
                               % 0-1 aralığındaki sayılardan kaça kadar
                               % olanı tumor var, kaça kadar olanı tümor
                               % yok anlamına geldiginiz belirliyor

%sgdm optimizer algoritmasının adı.. diger algoritmaları da
%deneyebilirsiniz. adam, rmsprop, lbfgs

options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(imdsTrain,layers,options);   % verisetine uygun trainingi başlatır.


YPred = classify(net,imdsTest);


YTest = imdsTest.Labels;

figure;
cm = confusionchart(YTest,YPred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
%https://en.wikipedia.org/wiki/Sensitivity_and_specificity


accuracy = sum(YPred == YTest)/numel(YTest)


analyzeNetwork(net);
% fc layerdaki 2880 lik flatten edilmiş array lazım
layer = 2;
name = net.Layers(layer).Name
channels = 1:20;
I = deepDreamImage(net,name,channels, ...
    'PyramidLevels',1);

figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')


% belirli bir image için fully connected layer çıktısı
% https://www.mathworks.com/help/deeplearning/ug/classify-videos-using-deep-learning.html
name = net.Layers(5).Name;
 
featureSet = activations(net,imdsTrain,net.Layers(4).Name,'OutputAs','columns');
% 33300 x6 ... 6 adet veri için 33300'lik array featureset olsu
% MRI veri seti ile eğitiğiniz model için MRI veri setinin bu şekilde
% flatten array dataset'ini alacaksınız.
% PET veri seti ile eğiteceğiniz ikinci bir model PET veri setinin flatten
% array datasetini alacaksınız.
% image-fusion : post-fusion (bu örnek post fusuion.. ayrı ayrı eğitip sonra birleştirir), pre-fusion (önce görüntüleri birleştirir ve tek image dataset ile tek model eğitir)
% iki array nasıl birleştirir: 2880 + 2880 = değerleri toplayıp 2880 yeni
% array oluşturursunuz ve ya uc uca ekleyip 5760 lık bir array
% oluşturursunuz.

% sonrası basit bir ML modeli eğitimi olacak.  ANN olur SVM olur regression
% olur farketmez.
% https://www.mathworks.com/help/deeplearning/ref/feedforwardnet.html



