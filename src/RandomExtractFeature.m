classdef RandomExtractFeature < ExtractFeature
    %RANDOMEXTRACTFEATURE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        resize_size = [12 12];
    end
    
    methods
        function [Ad] = extract(self, A)
            if size(A,3) > 1
                A = rgb2gray(A);
            end
            Ad = imresize(A, self.resize_size);
            Ad = Ad(:)';
        end
    end
    
end

