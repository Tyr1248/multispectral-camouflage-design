% 读取图像
img = imread("E:\ProjectX\Test_data_env_fig\Summer_camo.png");  % 替换为你的图像路径

% 获取图像尺寸
[h, w, c] = size(img);

% 1. 提取中心512x512区域
% 计算中心点
center_h = floor(h/2);
center_w = floor(w/2);

% 计算起始和结束索引
start_h = center_h - 255;  % 512/2 = 256，但起始索引是1
end_h = center_h + 256;    % 包含256个像素
start_w = center_w - 255;
end_w = center_w + 256;

% 提取中心区域
center_512 = zeros(512, 512, c, class(img));
for i = 1:512
    for j = 1:512
        for k = 1:c
            center_512(i, j, k) = img(start_h + i - 1, start_w + j - 1, k);
        end
    end
end

% 2. 下采样到256x256（使用平均池化）
result_256 = zeros(256, 256, c, class(img));

% 每2x2区域取平均
for i = 1:256
    for j = 1:256
        for k = 1:c
            % 计算2x2区域的平均值
            sum_val = 0;
            for di = 0:1
                for dj = 0:1
                    sum_val = sum_val + double(center_512(2*i-1+di, 2*j-1+dj, k));
                end
            end
            result_256(i, j, k) = cast(sum_val / 4, class(img));
        end
    end
end

% 显示结果
figure;
subplot(1,2,1); imshow(center_512); title('中心512x512区域');
subplot(1,2,2); imshow(result_256); title('下采样到256x256');
imwrite(result_256, 'Summer_camo_256.png');