variable "cloud_id" {
  type        = string
  description = "Yandex Cloud cloud_id."
}

variable "folder_id" {
  type        = string
  description = "Yandex Cloud folder_id where FlyBrain resources live."
}

variable "zone" {
  type        = string
  default     = "ru-central1-a"
  description = "Yandex Cloud availability zone."
}

variable "service_account_name" {
  type    = string
  default = "flybrain-sa"
}

variable "container_registry_name" {
  type    = string
  default = "flybrain-registry"
}

variable "s3_bucket_name" {
  type    = string
  default = "flybrain-data"
}
