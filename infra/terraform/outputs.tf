output "service_account_id" {
  value = yandex_iam_service_account.flybrain.id
}

output "container_registry_id" {
  value = yandex_container_registry.flybrain.id
}

output "s3_bucket" {
  value = yandex_storage_bucket.flybrain_data.bucket
}

output "s3_access_key_id" {
  value     = yandex_iam_service_account_static_access_key.flybrain_s3.access_key
  sensitive = true
}
