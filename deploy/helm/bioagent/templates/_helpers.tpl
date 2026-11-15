{{- define "bioagent.name" -}}{{ .Chart.Name }}{{- end -}}
{{- define "bioagent.fullname" -}}{{ printf "%s-%s" .Release.Name .Chart.Name | trunc 50 | trimSuffix "-" }}{{- end -}}
{{- define "bioagent.labels" -}}
app.kubernetes.io/name: {{ include "bioagent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}
{{- define "bioagent.selector" -}}
app.kubernetes.io/name: {{ include "bioagent.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}
{{- define "bioagent.redisUrl" -}}
{{- if .Values.redis.enabled -}}redis://{{ include "bioagent.fullname" . }}-redis:6379/0{{- else -}}{{ required "redis.externalUrl is required when redis.enabled=false" .Values.redis.externalUrl }}{{- end -}}
{{- end -}}
{{- define "bioagent.secretName" -}}
{{- if .Values.secrets.create -}}{{ include "bioagent.fullname" . }}-secrets{{- else -}}{{ .Values.secrets.existingSecret }}{{- end -}}
{{- end -}}
{{- define "bioagent.image" -}}{{ .Values.image.repository }}:{{ .Values.image.tag }}{{- end -}}
{{/* Shared env, volumes and security settings for the api and worker pods. */}}
{{- define "bioagent.env" -}}
envFrom:
  - configMapRef:
      name: {{ include "bioagent.fullname" . }}-config
  - secretRef:
      name: {{ include "bioagent.secretName" . }}
      optional: true
{{- end -}}
{{- define "bioagent.volumeMounts" -}}
volumeMounts:
  - {name: data, mountPath: /app/data}
  - {name: results, mountPath: /app/results}
  - {name: hf-cache, mountPath: /app/.hf_cache}
  - {name: tmp, mountPath: /tmp}
{{- end -}}
{{- define "bioagent.volumes" -}}
volumes:
  - name: data
  {{- if and .Values.persistence.enabled .Values.persistence.hostPath }}
    hostPath: {path: {{ .Values.persistence.hostPath }}, type: DirectoryOrCreate}
  {{- else if .Values.persistence.enabled }}
    persistentVolumeClaim:
      claimName: {{ .Values.persistence.existingClaim | default (printf "%s-data" (include "bioagent.fullname" .)) }}
  {{- else }}
    emptyDir: {}
  {{- end }}
  - {name: results, emptyDir: {}}
  - {name: hf-cache, emptyDir: {}}
  - {name: tmp, emptyDir: {}}
{{- end -}}
