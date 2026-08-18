% Concise version - run directly
clc; clear; close all;

% 1. Input RGB values
rgb = [80 189 80];  % Modify to the desired RGB values here, e.g. [0 255 0] green, [0 0 255] blue

% 2. Clamp RGB values to the 0-255 range
rgb = max(0, min(255, rgb));

% 3. Create a 256x256 solid-color image
img = zeros(256, 256, 3, 'uint8');
img(:,:,1) = rgb(1);  % R channel
img(:,:,2) = rgb(2);  % G channel
img(:,:,3) = rgb(3);  % B channel

% 4. Display the image
imshow(img);
title(sprintf('单色图像 - RGB: [%d %d %d]', rgb(1), rgb(2), rgb(3)));

% 5. Save the image
filename = sprintf('solid_color_R%d_G%d_B%d.png', rgb(1), rgb(2), rgb(3));
imwrite(img, filename);
fprintf('图像已保存: %s\n', filename);