import pandas as pd

df = pd.read_csv('outputs/multiseed/gradient_cosine_raw.csv')

print('\nPer-seed gradient cosine mean:')
print('='*70)
for feat in sorted(df['feature'].unique()):
    sub = df[df['feature']==feat]
    print(f'\n  {feat}:')
    seed_means = sub.groupby('seed')['grad_cosine_aff_spec'].mean()
    for seed, val in seed_means.items():
        print(f'    Seed {seed:>5}: {val:+.4f}')
    print(f'    --------------------------------')
    print(f'    Mean of seeds: {seed_means.mean():+.4f} +/- {seed_means.std():.4f}')
    print(f'    Range: [{seed_means.min():+.4f}, {seed_means.max():+.4f}]')
    all_positive = all(v > 0 for v in seed_means)
    print(f'    All seeds positive? {"YES" if all_positive else "NO"}')
