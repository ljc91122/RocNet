function [feas_all,label] = get_feas_vox(vox,k)
%collect feas in octree order, vox must be (k^n)^3, k is the length of the
%leaf vox, should be power of 2
%label: 0:leaf_full 1:leaf_empty 2:leaf_mix 3:interior(must be mix)

    n = size(vox,1);
    
%     if ~exist('base','var') || isempty(base)
%         base = [0 0 0];
%     end
    
    if n<k
        error('dim must be larger than k');
    end
    
    if all(vox,'all')
        feas_all = vox;
        label = 0;
        return;
    end
    
    if all(~vox,'all')
        feas_all = zeros(k,k,k);
        label = 1;
        return;
    end
    
    if n==k
        feas_all = vox;
        label = 2;
        return;
    end
    
    
    [feas1,l1] = get_feas_vox(vox(1:floor(n/2),1:floor(n/2),1:floor(n/2)),k);
    [feas2,l2] = get_feas_vox(vox(1+floor(n/2):n,1:floor(n/2),1:floor(n/2)),k);
    [feas3,l3] = get_feas_vox(vox(1:floor(n/2),1+floor(n/2):n,1:floor(n/2)),k);
    [feas4,l4] = get_feas_vox(vox(1+floor(n/2):n,1+floor(n/2):n,1:floor(n/2)),k);
    
    [feas5,l5] = get_feas_vox(vox(1:floor(n/2),1:floor(n/2),1+floor(n/2):n),k);
    [feas6,l6] = get_feas_vox(vox(1+floor(n/2):n,1:floor(n/2),1+floor(n/2):n),k);
    [feas7,l7] = get_feas_vox(vox(1:floor(n/2),1+floor(n/2):n,1+floor(n/2):n),k);
    [feas8,l8] = get_feas_vox(vox(1+floor(n/2):n,1+floor(n/2):n,1+floor(n/2):n),k);
    
    feas_all = cat(4,feas1,feas2,feas3,feas4,feas5,feas6,feas7,feas8);
    label = [l1;l2;l3;l4;l5;l6;l7;l8;3];
    
end