% 简洁版本 - 直接运行
clc; clear; close all;

% 1. 输入RGB值
rgb = [80 189 80];  % 这里可以修改为你想要的RGB值，例如：[0 255 0]绿色，[0 0 255]蓝色

% 2. 确保RGB值在0-255之间
rgb = max(0, min(255, rgb));

% 3. 创建256x256单色图像
img = zeros(256, 256, 3, 'uint8');
img(:,:,1) = rgb(1);  % R通道
img(:,:,2) = rgb(2);  % G通道
img(:,:,3) = rgb(3);  % B通道

% 4. 显示图像
imshow(img);
title(sprintf('单色图像 - RGB: [%d %d %d]', rgb(1), rgb(2), rgb(3)));

% 5. 保存图像
filename = sprintf('solid_color_R%d_G%d_B%d.png', rgb(1), rgb(2), rgb(3));
imwrite(img, filename);
fprintf('图像已保存: %s\n', filename);