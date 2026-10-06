// Shapes of the artifact contract (§6). The UI renders these generically.

export interface WorkflowNode { id: string; label: string; detail: string; formula_ref: string | null }
export interface Glossary { sym: string; means: string }
export interface Formula { id: string; name?: string; latex: string; caption: string; glossary: Glossary[] }
export interface Assumption { text: string; status: 'ok' | 'warn' | 'fail' | 'required' | 'unknown'; evidence: string }
export interface FigureRef { file: string; title: string; caption: string; type: string; page?: string }
export interface Extra { kind: string; file: string; title: string }

export interface Meta {
  model_id: string
  display_name: string
  family: string
  task: string
  one_liner: string
  plain_explanation: string
  year: number
  reference: string
  workflow: WorkflowNode[]
  edges: [string, string][]
  formulas: Formula[]
  assumptions: Assumption[]
  figures: FigureRef[]
  extras?: Extra[]
}

export interface RunMetrics {
  primary?: boolean
  threshold?: number
  accuracy: number; precision: number; recall: number; f1: number
  roc_auc: number; pr_auc: number; mcc: number; brier?: number
  cv_roc_auc_mean?: number | null; cv_roc_auc_std?: number | null
  confusion_matrix?: number[][]
  train_seconds?: number; predict_seconds?: number
  capture_at_20?: number
}

export interface Metrics {
  model_id: string
  task?: string
  protocol?: Record<string, unknown>
  runs?: { without_duration: RunMetrics; with_duration: RunMetrics }
  baselines?: { majority_class: { accuracy: number; roc_auc: number; pr_auc: number } }
  leak_delta?: { roc_auc: number; pr_auc: number; f1: number }
  [k: string]: unknown
}

export interface ConfigParam { name: string; value: unknown; why: string }
export interface ModelConfig { estimator: string; params: ConfigParam[] }

export interface Finding { type: 'result' | 'limitation' | 'failure'; text: string }
export interface Findings { headline: string; findings: Finding[]; comparison_note: string; viva_answer: string }

export interface Narrative { headline: string; body: string; caveat: string }

export interface ShapExplain {
  kind: 'shap'
  model_id: string
  explainer: string
  sample_size: number
  units: string
  base_value: number
  global: { features: string[]; mean_abs_shap: number[]; share: number[]; direction: string[]; direction_text: string[]; macro_share: number; plain_text: string }
  beeswarm: { features: string[]; values: number[][]; feature_values: (number | null)[][]; labels: string[][] }
  local_examples: { client_id: number; label: string; prediction: number; y_true: number; base_value: number; contributions: { feature: string; value: number }[]; plain_text: string }[]
  narrative: Narrative
  mask_vs_shap?: { spearman_rho: number; top5_overlap: string[] }
}

export interface ClusterExplain {
  kind: 'cluster_profile'
  clusters: { id: number; name: string; size: number; subscribe_rate: number; profile: string; deviations: { feature: string; kind: string; deviation: number; text: string }[] }[]
  global: { plain_text: string }
  narrative: Narrative
}

export interface RuleExplain {
  kind: 'rule_contribution'
  items: { item: string; plain_text: string; rules: { rule: string; lift: number; confidence: number; support: number; coverage: number }[] }[]
  global: { plain_text: string }
  narrative: Narrative
}

export type Explain = ShapExplain | ClusterExplain | RuleExplain

export interface ManifestModel { model_id: string; display_name?: string; family?: string; task?: string; one_liner?: string; status: 'ok' | 'invalid' | 'missing'; errors?: string[] }
export interface Manifest { project: string; generated: string; dataset_sha256: string; dataset_ok: boolean; models: ManifestModel[]; comparison_ok: boolean }

export interface Curves {
  roc: { fpr: number[]; tpr: number[] }
  pr: { recall: number[]; precision: number[] }
  lift: { fraction_called: number[]; fraction_captured: number[] }
}
