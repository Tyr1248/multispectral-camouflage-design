clc;
clear;
close all;

%% 设置路径
maskImagePath = "E:\ProjectX\Evaluation of Camo\gbvs-master\Arctic\arctic_mask.png";
saliencyImagePath = "E:\ProjectX\Evaluation of Camo\gbvs-master\arctic__amorphous_pattern_final_singlecolor_Saliency\arctic__amorphous_pattern_final_singlecolor_Itti_map.png";

%% 读取显著性图
[saliencyImage, ~] = imread(saliencyImagePath);
if size(saliencyImage, 3) == 3
    saliencyImage = rgb2gray(saliencyImage);
end
saliencyImage = double(saliencyImage);

%% 读取并安全转换掩膜图像为二值图
[maskImage, map] = imread(maskImagePath);

if ~isempty(map)
    % 索引图像
    if exist('im2gray', 'file') % R2020a+
        grayMask = im2gray(maskImage, map);
    else
        grayMask = rgb2gray(ind2rgb(maskImage, map));
    end
elseif size(maskImage, 3) == 3
    % RGB 图像
    grayMask = rgb2gray(maskImage);
else
    % 灰度或二值图像
    grayMask = maskImage;
end

% 转为二值图像
maskBinary = imbinarize(grayMask);

% 检查是否有前景
if ~any(maskBinary(:))
    error('掩膜图像中没有检测到任何前景像素！');
end

%% 标记连通区域
labeledMask = bwlabel(maskBinary);
stats = regionprops(labeledMask, 'BoundingBox', 'PixelIdxList');

if isempty(stats)
    error('未找到任何连通区域。');
end

nRegions = length(stats);

%% 预分配结果结构体（关键：避免字段不一致错误）
template = struct('AvgSaliency', 0.0, 'MaxSaliency', 0.0, 'BoundingBox', zeros(1,4));
results = repmat(template, nRegions, 1);

%% 可视化：在显著性图上绘制区域和标签
figure('Visible', 'on');  % 确保 figure 显示
imshow(saliencyImage, []);
hold on;

for idx = 1:nRegions
    pixelIdxList = stats(idx).PixelIdxList;
    sal_vals = saliencyImage(pixelIdxList);
    
    avgSaliency = mean(sal_vals);
    maxSaliency = max(sal_vals);
    
    % 自动归一化（如果显著性图是 uint8 范围 0-255）
    if max(saliencyImage(:)) > 1
        avgSaliency = avgSaliency / 255;
        maxSaliency = maxSaliency / 255;
    end
    
    results(idx).AvgSaliency = avgSaliency;
    results(idx).MaxSaliency = maxSaliency;
    results(idx).BoundingBox = stats(idx).BoundingBox;
    
    bb = stats(idx).BoundingBox;  % [x, y, w, h]
    rectangle('Position', bb, 'EdgeColor', 'r', 'LineWidth', 2);
    
    % 文本标签（防止越界）
    x_text = bb(1);
    y_text = max(bb(2) - 10, 10);
    text(x_text, y_text, ...
        sprintf('Avg: %.2f\nMax: %.2f', avgSaliency, maxSaliency), ...
        'Color', 'yellow', 'FontSize', 20, 'FontWeight', 'bold', ...
        'BackgroundColor', [0 0 0 0.6], 'VerticalAlignment', 'bottom');
end

title('Mask Regions with Normalized Saliency');
hold off;

%% 判断模型名称（用于文件命名）
modelName = 'GBVS';
if contains(saliencyImagePath, 'Itti', 'IgnoreCase', true)
    modelName = 'Itti';
end

%% 获取输出目录
folder = fileparts(saliencyImagePath);
if isempty(folder)
    folder = '.';
end

%% 1. 保存可视化图像
visFileName = fullfile(folder, ['saliency_visualization_' modelName '.png']);
visFileName = char(visFileName);  % 确保是 char 向量
print(gcf, '-dpng', visFileName);
disp(['✅ 可视化图像已保存至: ', visFileName]);

%% 2. 保存 CSV 结果
csvFileName = fullfile(folder, ['saliency_labels_' modelName '.csv']);
csvFileName = char(csvFileName);

T = table((1:nRegions)', ...
    [results.AvgSaliency]', ...
    [results.MaxSaliency]', ...
    'VariableNames', {'RegionID', 'AverageSaliency', 'MaximumSaliency'});

writetable(T, csvFileName);
disp(['✅ CSV 结果已保存至: ', csvFileName]);
