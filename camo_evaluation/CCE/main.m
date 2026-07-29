img = imread('camo.png');      % 原图
mask = imread('mask.png') ;  % 二值掩膜

% 转为逻辑掩膜（假设白色=目标）
if size(mask,3) == 3
    mask = rgb2gray(mask) > 128;
else
    mask = mask > 128;
end

mask = logical(mask);  % 确保是 logical 类型

[CCE, SL, SC, ST] = evaluateCamouflageEffect(img, mask);