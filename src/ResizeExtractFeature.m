classdef ResizeExtractFeature < ExtractFeature
    
    properties
        resize_size = [12 10];
    end
    
    methods
        function [A] = extract(self, images)
            self.img_size = self.resize_size;
            
            A = self.readImages(images);
        end
    end
    
end

