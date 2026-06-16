# FlyBrain Optimizer infrastructure on Yandex Cloud.
#
# Phase 0 ships the skeleton only — a Service Account with the roles needed to
# call YandexGPT, push images to Container Registry, and read/write S3 data.
# `terraform apply` is intentionally left out of CI; the user runs it once
# before Phase 7 (real LLM traces) or Phase 12.

resource "yandex_iam_service_account" "flybrain" {
  name        = var.service_account_name
  description = "FlyBrain Optimizer service account (LLM, registry, storage)."
  folder_id   = var.folder_id
}

locals {
  flybrain_sa_roles = [
    "ai.languageModels.user",
    "ai.embeddings.user",
    "container-registry.images.puller",
    "container-registry.images.pusher",
    "storage.editor",
    "datasphere.user",
  ]
}

resource "yandex_resourcemanager_folder_iam_member" "flybrain_sa" {
  for_each  = toset(local.flybrain_sa_roles)
  folder_id = var.folder_id
  role      = each.value
  member    = "serviceAccount:${yandex_iam_service_account.flybrain.id}"
}

resource "yandex_iam_service_account_static_access_key" "flybrain_s3" {
  service_account_id = yandex_iam_service_account.flybrain.id
  description        = "Static key for S3 access (connectome data, traces)."
}

resource "yandex_container_registry" "flybrain" {
  name      = var.container_registry_name
  folder_id = var.folder_id
}

resource "yandex_storage_bucket" "flybrain_data" {
  bucket     = var.s3_bucket_name
  folder_id  = var.folder_id
  access_key = yandex_iam_service_account_static_access_key.flybrain_s3.access_key
  secret_key = yandex_iam_service_account_static_access_key.flybrain_s3.secret_key
}
