from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

meter = metrics.get_meter('text_embedding_service')

model_inference_duration = meter.create_histogram(
    name='model_inference_duration_seconds',
    description='Time spent on model inference',
    unit='s'
)

model_embeddings_generated = meter.create_counter(
    name='model_embeddings_generated_total',
    description='Total number of embeddings generated',
    unit='1'
)

model_batch_size = meter.create_histogram(
    name='model_batch_size',
    description='Batch size used for encoding',
    unit='1'
)

model_memory_usage = meter.create_gauge(
    name='model_memory_usage_bytes',
    description='Memory usage by the model in bytes',
    unit='By'
)

task_manager_tasks = meter.create_counter(
    name='task_manager_tasks_total',
    description='Total number of tasks by status',
    unit='1'
)

task_manager_tasks_active = meter.create_up_down_counter(
    name='task_manager_tasks_active',
    description='Number of currently active tasks',
    unit='1'
)

task_manager_task_duration = meter.create_histogram(
    name='task_manager_task_duration_seconds',
    description='Time spent processing tasks',
    unit='s'
)

task_manager_texts_per_task = meter.create_histogram(
    name='task_manager_texts_per_task',
    description='Number of texts per task',
    unit='1'
)

task_manager_cleanup = meter.create_counter(
    name='task_manager_cleanup_total',
    description='Number of old tasks cleaned up',
    unit='1'
)

embedding_text_length = meter.create_histogram(
    name='embedding_text_length',
    description='Length of input texts',
    unit='1'
)

embedding_distance_matrix_size = meter.create_histogram(
    name='embedding_distance_matrix_size',
    description='Size of distance matrix (n x n)',
    unit='1'
)

process_cpu_seconds = meter.create_counter(
    name='process_cpu_seconds_total',
    description='Total user and system CPU time spent',
    unit='s'
)

process_memory = meter.create_gauge(
    name='process_resident_memory_bytes',
    description='Resident memory size in bytes',
    unit='By'
)

gpu_utilization = meter.create_gauge(
    name='gpu_utilization_percent',
    description='GPU utilization percentage',
    unit='%'
)


def setup_metrics():
    prometheus_reader = PrometheusMetricReader()

    provider = MeterProvider(
        metric_readers=[prometheus_reader]
    )

    metrics.set_meter_provider(provider)

    return prometheus_reader
