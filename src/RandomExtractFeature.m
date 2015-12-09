classdef RandomExtractFeature < ExtractFeature
    %RANDOMEXTRACTFEATURE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        resize_size = [12 10];
    end
    
    methods
        function [A] = extract(self, images)
            N = numel(images);
            A = zeros(N, prod(self.resize_size));
            for i = 1 : N
                img = images{i};
                A(i,:) = processInputImage(self, img);
            end
        end
        
        function I = processInputImage(self, img)
            if (size(img,3) > 1)
                img = rgb2gray(img);
            end
            img = imresize(img, self.resize_size);
            img = im2double(img);
            I = img(:)';
        end
    end
    
end

