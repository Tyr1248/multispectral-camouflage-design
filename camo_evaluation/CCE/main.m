img = imread('camo.png');      % original image
mask = imread('mask.png') ;  % binary mask

% Convert to a logical mask (white = target)
if size(mask,3) == 3
    mask = rgb2gray(mask) > 128;
else
    mask = mask > 128;
end

mask = logical(mask);  % ensure logical type

[CCE, SL, SC, ST] = evaluateCamouflageEffect(img, mask);