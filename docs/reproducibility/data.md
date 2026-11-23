# Data access

No media, features or identity manifests are distributed with this repository. Obtain each dataset yourself under its own terms.

| Dataset | Use in this project | Where | Notes |
|---|---|---|---|
| FLAG 2027 training set (MAV-Celeb v4 subset; 70 identities, 6,485 pairs, English) | training and all local validation | challenge Google Drive folder linked from the FLAG 2027 README / CodaBench competition 18062 | registration with the challenge |
| FLAG 2027 evaluation phase (English heard, Bangla unheard) | scoring only | same folder | Bengali speech may not be used to train or adapt audio encoders (challenge rules) |
| MAV-Celeb v1 (English + Urdu, 64 training identities) | external training data; Urdu holdout | github.com/mavceleb/mavceleb_baseline (Drive links) | training splits only |
| MAV-Celeb v3 (English + German, 50 training identities) | external training data; German holdout | same | training splits only |
| VoxCeleb2 test (117 identities after filtering) | external training data | VGG VoxCeleb (access form) | mp4; audio decoded at 16 kHz, a mid-clip frame as face |

**Processing applied** (`FLAG_2027/external_data/`):
- frozen VoxLingua107 language ID: any clip that looks Bengali is dropped;
- face-outlier removal against each identity's FaRL centroid;
- gender per identity: metadata, otherwise face and voice estimates, with manual overrides;
- duplicate removal against the official identities and merging across sources;
- 231 external identities and 29,702 pairs in total.

**Fold definitions** in `validation/` are row indices and anonymised challenge identity codes (`id001`…) derived from the official training manifest. They contain no media.
