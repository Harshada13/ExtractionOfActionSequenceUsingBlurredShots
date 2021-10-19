function []=main()
%imageList = {'Image3.png' 'Image2.png' };
%imageList = {'s31.jpg' 's32.jpg' 's33.jpg'}; 
img1 = imread ("s11.jpg");
img2 = imread ("s12.jpg");
img3 = imread ("s13.jpg");
img4 = imread ("s14.jpg");
images = {img1 img2 img3 img4};
numObjects = 1; %foreground objects interested in
numImages = size(images,2);
originalSize = [1000 1000]; %Resolution
imageScale = 0.45;
resizedImages = cell(numImages,1);

for i = 1:numImages    
    resizedImages{i} = {imresize((cell2mat(images(i))), imageScale)};  
end

%Start by trying to find background from first and last image
baseImage = cell2mat(resizedImages{1});
newImage = cell2mat(resizedImages{numImages});
setBackground = true; %Should be true for first/last image comparison, false when just adding in a new foreground
background = extractbackground(baseImage, newImage, numObjects, setBackground);

%Add foreground of each image into compiled
foregroundMasks = cell(numImages,1);
backgroundMasks = cell(numImages,1);
setBackground = false;        %false when just adding new foregrounds
for i = 1:numImages
    baseImage = background;
    newImage =cell2mat(resizedImages{i});
    [foregroundMasks{i},backgroundMasks{i}] = foregroundEdge(baseImage, newImage, numObjects, setBackground);
    
end
blended = blending(background, foregroundMasks, resizedImages);

%Display image and background
figure('name', 'Background Detected');
imshow(background);
figure('name', 'Images Stritched'); 
imshow(blended);


 [foregroundMask,backgroundMask] = foregroundEdge(baseImage, newImage, numObjects, setBackground);
% Takes two rgb images with same background but moving foreground and
% returns foreground of second image and combined background of both images
% Last dilation factor should be 0 if background image has not been
% generated yet
hsv1 = rgb2hsv(baseImage);
h1 = hsv1(:,:,1);
hsv2 = rgb2hsv(newImage);
h2 = hsv2(:,:,1);

% Find differences between first and second images in hsv
normThresholded = hsvDiff(hsv1, hsv2);
%figure('name','HSV Thresholded image');
% imshow(normThresholded)

% Remove noise of differences by eroding and filtering small areas
numareas = numObjects;
filteredMask = nregions(normThresholded, numareas);

% Use edge correlation to differentiate foreground/background
if setBackground
    foregroundMask = foregroundextract(filteredMask, h1, h2);
else
    foregroundMask = filteredMask;
end
backgroundMask = filteredMask - foregroundMask;

normThreshold = hsvDiff(hsv1, hsv2)

difference1 = abs(hsv2-hsv1);
%Hue is a continuous scale, max diff is .5, treat all the differences around circle
huediff = difference1(:,:,1);
for i=1:size(huediff,1)
    for j=1:size(huediff,2)
        if huediff(i,j)>0.5
            huediff(i,j)= -huediff(i,j)+1;
        end
    end
end
huepart=zeros([size(huediff,1) size(huediff,2)]);
for i=1:size(huediff,1)
    for j=1:size(huediff,2)
        huepart(i,j)=huediff(i,j)*2;
    end
end

difference1(:,:,1) = huepart;

%Treat hsv channels as a vector, find norm of difference vectors for each
%pixel
normDifference = sqrt(difference1(:,:,1).^2 + difference1(:,:,2).^2 + difference1(:,:,3).^2);
maxvalue=max(max(normDifference));
normDifference = normDifference./maxvalue; %Get everything between 0-1

%figure;
%imshow(normDiffh);

smallval = 1e-8; %To avoid median threshold being exactly zero
medval = median(reshape(normDifference,1,numel(normDifference))) + smallval;
mulfactor = 3.5;  %Arbitrarily chosen for now
normThresh = normDifference >= medval*mulfactor;
%disp(normThresh)

SE2 = strel('disk',2); %Erode and re-dilate to get rid of small noisy regions
normThreshold = imopen(normThresh, SE2);

filteredMask = nregions(normThresholded, numAreas)

dilated = normThresholded;

%Pick out the largest regions to keep in the mask
[labels,numlabels] = bwlabel(dilated, 8); %Label by regions connected by 8 neighbors
disp(numlabels);
regions = regionprops(dilated,'Area');
areas = sortrows([(1:size(regions,1))' [(regions(:).Area)]'],2);
imgmask = zeros(size(labels));
if numlabels > 0
    for i = 1:min([numAreas numlabels])
        imgmask(labels == areas(end-i+1,1)) = 1;
    end
end
%figure('name','Image Mask');
%imshow(imageMask);
eroded = imgmask;

%Get rid of holes in the mask
SE3 = strel('disk', 5); 
dilated2 = imdilate(eroded, SE3);
filled = imfill(dilated2, 'holes');
filteredMask = imerode(filled, SE3);

%figure('name','filtered mask');
%imshow(filteredMask);

foregroundmask = foregroundextract(filteredMask, h1, h2)

SE1 = strel('disk', 1); % structuring elements for dilation/erosion
SE3 = strel('disk', 3); 
SE30 = strel('disk', 25);

% Find edges of hue channel
edge1 = edge(h1,'canny',[],3);
%figure('name','Background Edge Detection');
%imshow(edgeh1); 
edge2 = edge(h2,'canny',[],3);
%figure('name','Last Image Edge Detection');
%imshow(edgeh2);

% For each area in the filtered mask find correlation to combined edges
% Regions with much better correlation to 2nd image are part of foreground
% Regions with similar correlation need  to split overlapping foregrounds
[regionLabels,numAreas] = bwlabel(filteredMask,8);
correlation = [zeros(numAreas,1) zeros(numAreas,1) (1:numAreas)']; %Allocate memory
correlationThreshold = 3; %Arbitrarily choose 3x correlation
foregroundmask = zeros(size(filteredMask)); %Allocate memory

for i = 1:numAreas    
    filter = edge((regionLabels == i), 'canny'); % edge dilation    
    filterdil = imdilate(filter, SE3);
    %figure('name','Dilated filter edge');
    %imshow(filterEdgeD3);
    
    combine1 = ((edge1 & filterdil) & ~(imdilate(edge2,SE1) & filterdil));   % Overlap the image edges with filter edged
    %figure('name','combined edge 1');
    %imshow(combinedEdges1);
    combine2 = ((edge2 & filterdil) & ~(imdilate(edge1,SE1) & filterdil));
    %figure('name','combined edge 2');
    %imshow(combinedEdges2);
    correlation(i,1) = sum(combine1)/sum(filterdil);
    correlation(i,2) = sum(combine2)/sum(filterdil);

    if correlation(i,2) > correlation(i,1)*correlationThreshold   %Add to foreground
        foregroundmask(regionLabels == i) = 1; 
    elseif correlation(i,1) > correlation(i,2)*correlationThreshold %Don't add these to foreground
        
    else 
        temp1 = (combine1 & (regionLabels == i));
        temp2 = (combine2 & (regionLabels == i));
        close1 = imclose(temp1, SE30);
        close2 = imdilate(temp2, SE30);
        foregroundmask((close2 & ~close1) & (regionLabels == i)) = 1;
    end
end
% figure('name','pickforegrounds');
% imshow(foregroundmask);
