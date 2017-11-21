clear all;

lst1=strsplit(ls('leftImg8bit/train/*/*.png'));
lst2=strsplit(ls('leftImg8bit/val/*/*.png'));
lst = horzcat(lst1, lst2)

parfor i=1:numel(lst)-1
    disp(i)
    if(lst{i} ~= "")
	im=im2double(imread(lst{i}));
	im=imresize(im,[256 512]);
	if(size(im,3)==1)
	    im=repmat(im,[1 1 3]);
	end
	imwrite(im,sprintf('images/%d.png',i));
    end
end

lst3=strsplit(ls('gtFine/train/*/*color.png'));
lst4=strsplit(ls('gtFine/val/*/*color.png'));
images = horzcat(lst3, lst4)

parfor j=1:numel(images)-1
    if(images{j} ~= "")
	im=im2double(imread(strrep(strrep(images{j},'leftImg8bit.png','gtFine_color.png'),'leftImg8bit','gtFine')));
	if(length(size(im)) == 3)
	        disp(j)
		im=imresize(im,[256 512],'nearest');
		imwrite(im,sprintf('semantics/%d.png',j));
	end
    end
end
