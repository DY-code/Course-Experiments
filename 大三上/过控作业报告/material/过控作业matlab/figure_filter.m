%% matlab自带均值滤波器
% 结论：均值滤波半径越大，处理结果越模糊
clc, clear
% 读入图像
Image = imread('./待处理图片.png');
 
% 设置均值滤波
H3 = fspecial('average',[3,3]);
H5 = fspecial('average',[5,5]);
H7 = fspecial('average',[7,7]);
H9 = fspecial('average',[9,9]);
H11 = fspecial('average',[11,11]);
 
% 利用滤波对图像进行处理
r3 = imfilter(Image,H3);
r5 = imfilter(Image,H5);
r7 = imfilter(Image,H7);
r9 = imfilter(Image,H9);
r11 = imfilter(Image,H11);

% 展示结果
subplot(2,3,1);imshow(Image);title('原图');
subplot(2,3,2);imshow(r3);title('3*3均值滤波结果');
subplot(2,3,3);imshow(r5);title('5*5均值滤波结果');
subplot(2,3,4);imshow(r7);title('7*7均值滤波结果');
subplot(2,3,5);imshow(r9);title('9*9均值滤波结果');
subplot(2,3,6);imshow(r11);title('11*11均值滤波结果');

%% 分别给干净图像添加高斯和椒盐噪声，然后进行均值滤波、高斯滤波和中值滤波
% 结论：含高斯噪声的图像用高斯滤波处理时效果好，含椒盐噪声的图像用中值滤波处理时效果好
% 读入图像
Image = imread('./待处理图片.png');
 
% 给原图加入高斯噪声
GaussainI = imnoise(Image,'gaussian');
 
% 给原图加入椒盐噪声
SaltPepperI = imnoise(Image,'salt & pepper');
 
% 设置均值滤波
aveFilter3 = fspecial('average',[3,3]);
 
% 设置高斯滤波
gausFilter3 = fspecial('gaussian',[3,3],0.8);
 
tempG=rgb2gray(GaussainI);  %灰度处理，灰度处理后的图像是二维矩阵
tempSP=rgb2gray(SaltPepperI);
 
% 用均值滤波对高斯噪声图像进行处理
GJ = imfilter(GaussainI,aveFilter3);
 
% 用高斯滤波对高斯噪声图像进行处理
GG = imfilter(GaussainI,gausFilter3,'conv');
 
% 用中值滤波对高斯图像进行处理
GM = medfilt2(tempG,[3,3]);
 
% 用均值滤波对椒盐噪声图像进行处理
SPJ = imfilter(SaltPepperI,aveFilter3);
 
% 用高斯滤波对椒盐噪声图像进行处理
SPG = imfilter(SaltPepperI,gausFilter3,'conv');
 
% 用中值滤波对椒盐噪声图像进行处理
SPM = medfilt2(tempSP,[3,3]);
 
%展示结果
subplot(3,3,1);imshow(Image);title('原图');
subplot(3,3,2);imshow(GaussainI);title('添加高斯噪声后的图像');
subplot(3,3,3);imshow(SaltPepperI);title('添加椒盐噪声后的图像');
subplot(3,3,4);imshow(GJ);title('高斯噪声经均值滤波处理后');
subplot(3,3,5);imshow(GG);title('高斯噪声经高斯滤波处理后');
subplot(3,3,6);imshow(GM);title('高斯噪声经中值滤波处理后');
subplot(3,3,7);imshow(SPJ);title('椒盐噪声经均值滤波处理后');
subplot(3,3,8);imshow(SPG);title('椒盐噪声经高斯滤波处理后');
subplot(3,3,9);imshow(SPM);title('椒盐噪声经中值滤波处理后');
 
%%  中值滤波
clc, clear
% 读入图像
Image = imread('./待处理图片.png');

% 转换为灰度图
Image = rgb2gray(Image);

r3 = median_filter(Image, 3);
r5 = median_filter(Image, 5);
r7 = median_filter(Image, 7);
r9 = median_filter(Image, 9);
r11 = median_filter(Image, 11);

% imshow(uint8(r3));

% 展示结果
subplot(2,3,1);imshow(uint8(Image));title('原图');
subplot(2,3,2);imshow(uint8(r3));title('3*3均值滤波结果');
subplot(2,3,3);imshow(uint8(r5));title('5*5均值滤波结果');
subplot(2,3,4);imshow(uint8(r7));title('7*7均值滤波结果');
subplot(2,3,5);imshow(uint8(r9));title('9*9均值滤波结果');
subplot(2,3,6);imshow(uint8(r11));title('11*11均值滤波结果');















