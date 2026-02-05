/**
 * TypeScript type definitions for ObjDet API
 * These types match the OpenAPI specification and can be shared with backend
 */

// ============ Training Types ============

export interface TrainingConfig {
    name: string;
    model_architecture: 'yolov8' | 'yolov11' | 'retinanet' | 'fcos';
    dataset: string;
    epochs: number;
    batch_size?: number;
    learning_rate?: number;
    optimizer?: 'adam' | 'sgd' | 'adamw';
    gpu?: 'auto' | 'gpu0' | 'gpu1' | 'multi';
    mixed_precision?: 'fp16' | 'fp32' | 'bf16';
    save_checkpoints?: boolean;
    early_stopping?: boolean;
    log_to_mlflow?: boolean;
    data_augmentation?: boolean;
}

export interface TrainingJobResponse {
    task_id: string;
    created_at: string;
}

export interface TaskStatus {
    task_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress?: number;
    current_epoch?: number;
    total_epochs?: number;
}

export interface ActiveTask {
    task_id: string;
    name: string;
    status: string;
    progress?: number;
    current_epoch?: number;
    total_epochs?: number;
}

export interface ActiveTasksResponse {
    tasks: ActiveTask[];
}

// ============ MLFlow Types ============

export interface Experiment {
    experiment_id: string;
    name: string;
    artifact_location?: string;
}

export interface ExperimentsResponse {
    experiments: Experiment[];
}

export interface Run {
    run_id: string;
    run_name: string;
    status: string;
    experiment_id: string;
    start_time?: number;
}

export interface RunsResponse {
    runs: Run[];
}

export interface RunDetails {
    run_id: string;
    run_name: string;
    status: string;
    params?: Record<string, string>;
    metrics?: Record<string, number>;
}

export interface Metric {
    step: number;
    metric: string;
    value: number;
}

export interface MetricsResponse {
    metrics: Metric[];
}

export interface Artifact {
    path: string;
    is_dir: boolean;
    file_size?: number;
}

export interface ArtifactsResponse {
    artifacts: Artifact[];
}

// ============ System Types ============

export interface ServiceStatus {
    status: 'connected' | 'disconnected';
}

export interface SystemStatus {
    services: {
        celery: ServiceStatus;
        mlflow: ServiceStatus;
    };
}

// ============ API Client Types ============

export interface ListRunsParams {
    experimentId?: string;
    status?: string;
    maxResults?: number;
}
