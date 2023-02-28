clc;
clear;
load net;
[filename,pathname]=uigetfile('*.*','Pick a iamge');
I=imread(strcat(pathname.filename));
I=imresize(I,[720 960]);
C=semanticseg(I,net);
B=labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure;
imshow();
figure;
imshow(B)
pixelLabelColorbar(cmap,classes);