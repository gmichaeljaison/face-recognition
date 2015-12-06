classdef (Abstract) ExtractFeature < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        img_size = [60 60];
    end
    
    methods(Abstract)
        Ad = extract(A)
    end
    
    methods
        function A = readImages(self, images)
            if ~iscell(images)
                images = {images};
            end
            
            N = numel(images);
            A = zeros(N, prod(self.img_size));
            for i = 1 : N
                img = images{i};
                A(i,:) = processInputImage(self, img);
            end
        end
        
        function I = processInputImage(self, img)
            if (size(img,3) > 1)
                img = rgb2gray(img);
            end
            img = imresize(img, self.img_size);
%             img = im2double(img);
            I = img(:)';
        end
    end
    
end

