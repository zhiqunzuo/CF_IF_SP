data {
    int<lower = 0> N; // number of observation
    int<lower = 0> K; // number of covariates
    matrix[N, K]   a; // sensitive attributes
    array[N] real  ugpa; // UGPA
    array[N] real   lsat; //LSAT
    array[N] real  zfya; // ZFYA
}

transformed data {
    vector[K] zero_K;
    vector[K] one_K;

    zero_K = rep_vector(0, K);
    one_K = rep_vector(1, K);
}

parameters {
    vector[N] u;

    real ugpa0;
    real eta_u_ugpa;
    real lsat0;
    real eta_u_lsat;
    real eta_u_zfya;

    vector[K] eta_a_ugpa;
    vector[K] eta_a_lsat;
    vector[K] eta_a_zfya;

    real<lower = 0> sigma_g_Sq_gpa;
}

model {
    u ~ uniform(-2, 2);

    ugpa0 ~ normal(0, 1);
    eta_u_ugpa ~ normal(0, 1);
    lsat0 ~ normal(0, 1);
    eta_u_lsat ~ normal(0, 1);
    eta_u_zfya ~ normal(0, 1);

    eta_a_ugpa ~ normal(zero_K, one_K);
    eta_a_lsat ~ normal(zero_K, one_K);
    eta_a_zfya ~ normal(zero_K, one_K);

    //ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, 0.00001);
    //lsat ~ normal(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat), 0.00001);
    //zfya ~ normal(eta_u_zfya * u + a * eta_a_zfya, 0.00001);
    ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, 0.1);
    lsat ~ normal(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat), 0.1);
    zfya ~ normal(eta_u_zfya * u + a * eta_a_zfya, 0.1);
}