function y = gaussian_2d(x, sigma, centers)
   c1 = centers(1);
   c2 = centers(2);
   
   x1 = x(:,1);
   x2 = x(:,2);
   y = exp(-((x1-c1).^2 + (x2-c2).^2) / (2*sigma^2));
end