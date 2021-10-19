function blended = blending(background, foregroundMasks, resizedImages)
SEcompile = strel('disk', 50); 
blended = background;
prevmask = zeros(size(foregroundMasks{1}));

for i = 1:length(resizedImages)
    newimg = cell2mat(resizedImages{i});
    dilmask = imdilate(foregroundMasks{i}, SEcompile) - foregroundMasks{i};
    newmask = foregroundMasks{i} | (dilmask & ~prevmask);
    
    blended(:,:,1) = newimg(:,:,1).*uint8(newmask) + blended(:,:,1).*uint8(~newmask);  %Blend the image with the new foreground
    blended(:,:,2) = newimg(:,:,2).*uint8(newmask) + blended(:,:,2).*uint8(~newmask);
    blended(:,:,3) = newimg(:,:,3).*uint8(newmask) + blended(:,:,3).*uint8(~newmask);
    tempmask=foregroundMasks{i};
    prevmask = prevmask | tempmask;
end

end
