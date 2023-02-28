clc;
clear
%load vgg
vgg16();
%creating adatastore
imds=imageDatastore('701-stillsRaw_full');
disp(imds);
I=readimage(imds,1);
I=histeq(I);
imshow(I);
%create a data store for original images and labeled images
%load the classes
classes=["Sky","Building","Pole","Road","Pavement","Tree","SignSymbol","Fence","Car","Pedestrian","Bicycleist"];
labelIDs=camvidPixelLabelIDs();
labelDir=fullfile('LabeledApproved_full');
%pixel selection
pxds=pixelLabelDatastore(labelDir,classes,labelIDs);
C=readimage(pxds,1);
cmap=camvidColorMap;
B=labeloverlay(I,C,'Colormap',cmap);
imshow(B);
pixelLabelColorbar(cmap,classes);
tbl=countEachLabel(pxds);
%Analyzed dataset stastics
frequency=tbl.PixelCount/sum(tbl.PixelCount);
bar(1:numel(classes),frequency)
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
%Resize the dataset
imageFolder=fullfile('ResizedImage','imageResized',filesep);
imds=resizeCamVidIimages(imds,imageFolder);
labelFolder=fullfile('ResizedImage','labelsResized',filesep);
pxds=resizeCamVidPixelLabels(pxds,labelFolder);
%prepare training and testing set
[imdsTrain,imdsTest,pxdsTrain,pxdsTest]=partionCamVidData(imds,pxds);
numTrainingImages=numel(imdsTrain.Files);
numTestingImages = numel(imdsTest.Files)
%create a network
imageSize=[360 480 3];
numClasses=numel(classes);
igraph=segnetLayers(imageSize.numClasses,'vgg16');
%Balance the classes using weighting
imageFreq=tbl.pixelCount/tbl.imagePxelCount;
classWeights=median(imageFreq)/imageFreq
pxLayer=pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights)
igraph=removeLayers(igraph,'pixelLabels');
igraph=addLayers(igraph,pxLayer);
igraph=connectLayers(igraph,'softmax','labels');
%Data augmentation
augmenter=imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYReflection',[-10 10]);
pximds=pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);
%Training options
options=trainingOptions('sgdm',...
    'Momentum',0.9,...
    'InitialLearnRate',1e-3,...
    'L2Regularization',0.005,...
    'MaxEpochs',100,...
    'MiniBatchSize',2,...
    'Shuffle','every-epoch',...
    'VerboseFrequency',2);
%Train the network
[net,info]=trainNetwork(pximds,igraph,options);
save net net