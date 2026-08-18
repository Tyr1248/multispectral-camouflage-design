clc;
clear;
close all;

%% Set paths
maskImagePath = "E:\ProjectX\Evaluation of Camo\gbvs-master\Arctic\arctic_mask.png";
saliencyImagePath = "E:\ProjectX\Evaluation of Camo\gbvs-master\arctic__amorphous_pattern_final_singlecolor_Saliency\arctic__amorphous_pattern_final_singlecolor_Itti_map.png";

%% Read the saliency map
[saliencyImage, ~] = imread(saliencyImagePath);
if size(saliencyImage, 3) == 3
    saliencyImage = rgb2gray(saliencyImage);
end
saliencyImage = double(saliencyImage);

%% Read the mask image and safely convert it to binary
[maskImage, map] = imread(maskImagePath);

if ~isempty(map)
    % indexed image
    if exist('im2gray', 'file') % R2020a+
        grayMask = im2gray(maskImage, map);
    else
        grayMask = rgb2gray(ind2rgb(maskImage, map));
    end
elseif size(maskImage, 3) == 3
    % RGB image
    grayMask = rgb2gray(maskImage);
else
    % grayscale or binary image
    grayMask = maskImage;
end

% Convert to a binary image
maskBinary = imbinarize(grayMask);

% Check that there is foreground
if ~any(maskBinary(:))
    error('掩膜图像中没有检测到任何前景像素！');
end

%% Label connected regions
labeledMask = bwlabel(maskBinary);
stats = regionprops(labeledMask, 'BoundingBox', 'PixelIdxList');

if isempty(stats)
    error('未找到任何连通区域。');
end

nRegions = length(stats);

%% Preallocate the result struct (key: avoids inconsistent-field errors)
template = struct('AvgSaliency', 0.0, 'MaxSaliency', 0.0, 'BoundingBox', zeros(1,4));
results = repmat(template, nRegions, 1);

%% Visualization: draw regions and labels on the saliency map
figure('Visible', 'on');  % make sure the figure is shown
imshow(saliencyImage, []);
hold on;

for idx = 1:nRegions
    pixelIdxList = stats(idx).PixelIdxList;
    sal_vals = saliencyImage(pixelIdxList);
    
    avgSaliency = mean(sal_vals);
    maxSaliency = max(sal_vals);
    
    % Auto-normalize (if the saliency map is uint8, range 0-255)
    if max(saliencyImage(:)) > 1
        avgSaliency = avgSaliency / 255;
        maxSaliency = maxSaliency / 255;
    end
    
    results(idx).AvgSaliency = avgSaliency;
    results(idx).MaxSaliency = maxSaliency;
    results(idx).BoundingBox = stats(idx).BoundingBox;
    
    bb = stats(idx).BoundingBox;  % [x, y, w, h]
    rectangle('Position', bb, 'EdgeColor', 'r', 'LineWidth', 2);
    
    % Text label (kept inside the image bounds)
    x_text = bb(1);
    y_text = max(bb(2) - 10, 10);
    text(x_text, y_text, ...
        sprintf('Avg: %.2f\nMax: %.2f', avgSaliency, maxSaliency), ...
        'Color', 'yellow', 'FontSize', 20, 'FontWeight', 'bold', ...
        'BackgroundColor', [0 0 0 0.6], 'VerticalAlignment', 'bottom');
end

title('Mask Regions with Normalized Saliency');
hold off;

%% Determine the model name (used for file naming)
modelName = 'GBVS';
if contains(saliencyImagePath, 'Itti', 'IgnoreCase', true)
    modelName = 'Itti';
end

%% Get the output directory
folder = fileparts(saliencyImagePath);
if isempty(folder)
    folder = '.';
end

%% 1. Save the visualization image
visFileName = fullfile(folder, ['saliency_visualization_' modelName '.png']);
visFileName = char(visFileName);  % make sure it is a char vector
print(gcf, '-dpng', visFileName);
disp(['✅ 可视化图像已保存至: ', visFileName]);

%% 2. Save the CSV results
csvFileName = fullfile(folder, ['saliency_labels_' modelName '.csv']);
csvFileName = char(csvFileName);

T = table((1:nRegions)', ...
    [results.AvgSaliency]', ...
    [results.MaxSaliency]', ...
    'VariableNames', {'RegionID', 'AverageSaliency', 'MaximumSaliency'});

writetable(T, csvFileName);
disp(['✅ CSV 结果已保存至: ', csvFileName]);
