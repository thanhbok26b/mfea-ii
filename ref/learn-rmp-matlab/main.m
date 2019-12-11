load('data/100.mat');

N = 50;
K = 2;
D = 50;

helper  = Helper();
subpops = helper.get_subpops(population, skill_factor, K, N);
vars    = [D, D];
models  = helper.get_models(subpops, D);

prob_matrix = helper.get_prob_matrix(subpops, models, D);

prob_matrix(1).data

x = [];
y = [];

for i=1:1000
  rmp = 0.001 * i;
  value = helper.log_likelihood(rmp, prob_matrix, K);
  x = [x, rmp];
  y = [y, value];
end

plot(x, y);

rmp_opt = fminbnd(@(x)helper.log_likelihood(x, prob_matrix, K), 0, 1)
