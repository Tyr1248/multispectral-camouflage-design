% example of how to call gbvs with default params



img = imread('E:\ProjectX\Evaluation of Camo\gbvs-master\Syria\syria2_syria_64x4a_final_singlecolor.png');
[filepath, name, ext] = fileparts('E:\ProjectX\Evaluation of Camo\gbvs-master\Syria\syria2_syria_64x4a_final_singlecolor.png');

% 创建保存文件夹
output_folder = [name '_Saliency'];
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 计算显著图
out_gbvs = gbvs(img);
out_itti = ittikochmap(img);

% 获取原始图像尺寸
[img_height, img_width, img_channels] = size(img);

% 直接保存原始图像（无需处理）
imwrite(img, fullfile(output_folder, [name '_Original_Image.png']));

% 调整显著图大小以匹配原始图像尺寸
gbvs_map_resized = imresize(out_gbvs.master_map_resized, [img_height, img_width]);
itti_map_resized = imresize(out_itti.master_map_resized, [img_height, img_width]);

% 保存GBVS显著图
imwrite(gbvs_map_resized, fullfile(output_folder, [name '_GBVS_map.png']));

% 保存Itti显著图
imwrite(itti_map_resized, fullfile(output_folder, [name '_Itti_map.png']));

% 使用与show_imgnmap完全相同的叠加方法
% 创建GBVS叠加图
gbvs_overlay = heatmap_overlay(img, gbvs_map_resized, 'jet');
imwrite(gbvs_overlay, fullfile(output_folder, [name '_GBVS_map_overlayed.png']));

% 创建Itti叠加图
itti_overlay = heatmap_overlay(img, itti_map_resized, 'jet');
imwrite(itti_overlay, fullfile(output_folder, [name '_Itti_map_overlayed.png']));

fprintf('所有图像已保存到文件夹: %s\n', output_folder);

% 完全复制的heatmap_overlay函数
function omap = heatmap_overlay( img , heatmap, colorfun )

if ( strcmp(class(img),'char') == 1 ) img = imread(img); end
if ( strcmp(class(img),'uint8') == 1 ) img = double(img)/255; end

szh = size(heatmap);
szi = size(img);

if ( (szh(1)~=szi(1)) | (szh(2)~=szi(2)) )
  heatmap = imresize( heatmap , [ szi(1) szi(2) ] , 'bicubic' );
end
  
if ( size(img,3) == 1 )
  img = repmat(img,[1 1 3]);
end
  
if ( nargin == 2 )
    colorfun = 'jet';
end
colorfunc = eval(sprintf('%s(50)',colorfun));

heatmap = double(heatmap) / max(heatmap(:));
omap = 0.8*(1-repmat(heatmap.^0.8,[1 1 3])).*double(img)/max(double(img(:))) + repmat(heatmap.^0.8,[1 1 3]).* shiftdim(reshape( interp2(1:3,1:50,colorfunc,1:3,1+49*reshape( heatmap , [ prod(size(heatmap))  1 ] ))',[ 3 size(heatmap) ]),1);
omap = real(omap);
end