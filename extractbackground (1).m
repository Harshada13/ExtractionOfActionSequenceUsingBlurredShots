function background = extractbackground(baseImage, newImage, numObjects, setBackground)

[backgroundmask] = foregroundcomp(baseImage, newImage, numObjects, setBackground);
%figure('name','Component 1 of Background');
background(:,:,1) = baseImage(:,:,1).*uint8(~backgroundmask) + newImage(:,:,1).*uint8(backgroundmask);
%imshow(background(:,:,1));
background(:,:,2) = baseImage(:,:,2).*uint8(~backgroundmask) + newImage(:,:,2).*uint8(backgroundmask);
%figure('name','Component 2 of Background');
%imshow(background(:,:,2));
background(:,:,3) = baseImage(:,:,3).*uint8(~backgroundmask) + newImage(:,:,3).*uint8(backgroundmask);
%figure('name','Component 3 of Background');
%imshow(background(:,:,3));

 function [backgroundmask] = foregroundcomp(baseImage, newImage, numObjects, setBackground)
hsv1 = rgb2hsv(baseImage);
h1 = hsv1(:,:,1);
hsv2 = rgb2hsv(newImage);
h2 = hsv2(:,:,1);
normThresholded = hsvDiff(hsv1, hsv2); % Find differences between first and last images in hsv
%figure('name','HSV Thresholded image');
% imshow(normThresholded)

% Remove noise of differences by eroding and filtering small areas
numareas = numObjects;
mask = nregions(normThresholded, numareas);

% edge correlation for differentiating the foreground/background
if setBackground
    foregroundMask = foregroundextract(mask, h1, h2);
else
    foregroundMask = mask;
end
backgroundmask = mask - foregroundMask;

function normThreshold = hsvDiff(hsv1, hsv2)

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

SE2 = strel('disk',2,0); %Erode and re-dilate to get rid of small noisy regions
normThreshold = imopen(normThresh, SE2);


function foregroundmask = foregroundextract(filteredMask, h1, h2)

SE1 = strel('disk', 1,0); % structuring elements for dilation/erosion
SE3 = strel('disk', 3,0); 
SE30 = strel('disk', 25,0);

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

function filteredMask = nregions(normThresholded, numAreas)

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