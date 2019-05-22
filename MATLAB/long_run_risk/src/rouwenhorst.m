function [y, P, s] = rouwenhorst(rho, sigma, n)

y_bar = sqrt((n - 1)/(1 - rho^2))*sigma;
y = linspace(-y_bar, y_bar, n);
p = (1 + rho)/2; 
q = p;
P = apply_rouwenhorst(n);
s=zeros(n, 1);
for j = 1:n
    s(j) = nchoosek(n - 1,j - 1);
end
s = s/2^(n - 1);

    function P = apply_rouwenhorst(h)
        if h == 2
            P = [p 1-p; 1-q q];
        else
            Pnew = apply_rouwenhorst(h - 1);
            z = zeros(1, h);
            z1 = zeros(h - 1, 1);
            P = [p s* Pnew z1; z]+[z1 (1 - p) * Pnew; z] + [z; (1-q) * Pnew z1]+[z; z1 q * Pnew];
            P(2:h - 1 ,:) = P(2:h - 1 ,:)/2;
        end
    end
end