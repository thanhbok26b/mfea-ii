classdef Helper

  methods

    function subpops = get_subpops(self, population, skill_factor, K, N)
      for k = 1:K
        idx = find(skill_factor == k - 1);
        idx = idx(1:N);
        subpops(k).data = population(idx, :);
        fprintf('Subpop %d - %d individuals\n', k, length(subpops(k).data))
      end
    end

    function models = get_models(self, subpops, D)
      K = length(subpops);
      for k = 1:K
        models(k).num_sample = length(subpops(k).data);
        num_random_sample    = floor(0.2 * models(k).num_sample);
        random_pop           = rand(num_random_sample, D);
        models(k).mean       = mean([subpops(k).data; random_pop]); 
        models(k).stdev      = std([subpops(k).data; random_pop]); 
      end
    end

    function prob_matrix = get_prob_matrix(self, subpops, models, D)
      K = length(subpops);
      for k = 1:K-1
        for j = k+1:K
          prob_matrix(1).data = ones(models(k).num_sample, 2);
          prob_matrix(2).data = ones(models(j).num_sample, 2);
        end

        for i = 1:models(k).num_sample
          for l = 1:D
            prob_matrix(1).data(i, 1) = prob_matrix(1).data(i, 1) * pdf('Normal', subpops(k).data(i, l), models(k).mean(l),models(k).stdev(l));
            prob_matrix(1).data(i, 2) = prob_matrix(1).data(i, 2) * pdf('Normal', subpops(k).data(i, l), models(j).mean(l),models(j).stdev(l));
          end
        end

        for i = 1:models(j).num_sample
          for l = 1:D
            prob_matrix(2).data(i, 1) = prob_matrix(2).data(i, 1) * pdf('Normal', subpops(j).data(i, l), models(k).mean(l),models(k).stdev(l));
            prob_matrix(2).data(i, 2) = prob_matrix(2).data(i, 2) * pdf('Normal', subpops(j).data(i, l), models(j).mean(l),models(j).stdev(l));
          end
        end
      end
    end

    function value = log_likelihood(self, rmp, prob_matrix, K)
      value = 0;
      for k = 1:2
        for j = 1:2
          if k == j
            prob_matrix(k).data(:, j) = prob_matrix(k).data(:, j) * (1 - 0.5 * (K-1) * rmp / K);
          else
            prob_matrix(k).data(:, j) = prob_matrix(k).data(:, j) * 0.5 * (K-1) * rmp / K;
          end
        end
        value = value + sum(-log(sum(prob_matrix(k).data, 2)));
      end
    end

  end
end
