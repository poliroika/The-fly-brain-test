# Terraform — FlyBrain on Yandex Cloud

Phase 0 ships the skeleton. Real `apply` lands in Phase 12 (or earlier, before
Phase 7 if you need real LLM traces).

## Prerequisites

* Terraform `>= 1.5`
* Yandex Cloud CLI (`yc`) authenticated
* Cloud + folder IDs available as environment variables:

```bash
export YC_CLOUD_ID=$(yc config get cloud-id)
export YC_FOLDER_ID=$(yc config get folder-id)
```

## Layout

* `versions.tf` — provider pinning.
* `variables.tf` — `cloud_id`, `folder_id`, `zone`, optional name overrides.
* `main.tf` — service account + roles + Container Registry + S3 bucket.
* `outputs.tf` — IDs / bucket / S3 access keys (sensitive).

## Usage

```bash
cd infra/terraform
terraform init
terraform plan \
  -var "cloud_id=$YC_CLOUD_ID" \
  -var "folder_id=$YC_FOLDER_ID"

# When you are ready (Phase 12 or before Phase 7):
terraform apply \
  -var "cloud_id=$YC_CLOUD_ID" \
  -var "folder_id=$YC_FOLDER_ID"
```

## Note

Phase 0 does NOT run `terraform apply`. The repository ships only the
declarative spec; the user opts in explicitly before incurring any cloud
spend.
