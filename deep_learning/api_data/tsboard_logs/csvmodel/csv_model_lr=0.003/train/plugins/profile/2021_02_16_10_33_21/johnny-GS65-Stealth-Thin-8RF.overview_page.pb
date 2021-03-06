?	?F=D?aP@?F=D?aP@!?F=D?aP@	|?m?g??|?m?g??!|?m?g??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?F=D?aP@???
~N@1zVҊo(??A]???a??IM?J?@Y??WΦ?*	\???(`@2F
Iterator::Modelۤ???w??!Wb^??8G@)?h?^??1?
??O?8@:Preprocessing2U
Iterator::Model::ParallelMapV2????"2??!????i}5@)????"2??1????i}5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~31]??!:?
???8@)???????1Y?'g?V4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?S?4????!?I[??6@)??E?n???1ֶ?{^+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicehB?Ēr??!Bܰ??!@)hB?Ēr??1Bܰ??!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??0?*x?!?[?6k@)??0?*x?1?[?6k@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???V%???!???C#?J@)υ?^??w?1=J?=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%???}???!???%8@)??U?Z^?1T??u????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9|?m?g??I?N????X@Q?k?()??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???
~N@???
~N@!???
~N@      ??!       "	zVҊo(??zVҊo(??!zVҊo(??*      ??!       2	]???a??]???a??!]???a??:	M?J?@M?J?@!M?J?@B      ??!       J	??WΦ???WΦ?!??WΦ?R      ??!       Z	??WΦ???WΦ?!??WΦ?b      ??!       JGPUY|?m?g??b q?N????X@y?k?()???";
sequential_12/dense_36/MatMulMatMul{?4?d??!{?4?d??0"I
+gradient_tape/sequential_12/dense_36/MatMulMatMulu'>d??!x?"?d??0"I
+gradient_tape/sequential_12/dense_37/MatMulMatMulu'>d??!??RK??0";
sequential_12/dense_37/MatMulMatMul?????,??!?}w֧?0"I
-gradient_tape/sequential_12/dense_37/MatMul_1MatMul???cҽ??!>,??E??";
sequential_12/dense_38/MatMulMatMulˈ??z?!?Eh????0"Y
8gradient_tape/sequential_12/dense_37/BiasAdd/BiasAddGradBiasAddGrad3A???z?!6h??v??"I
-gradient_tape/sequential_12/dense_38/MatMul_1MatMul3A???z?!2I?
C ??"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch??(l?-v?!?Ҏ!???"M
4sequential_12/batch_normalization_24/batchnorm/mul_1Mul??(l?-v?!0\Q8????Q      Y@Y?? _??@af?*??W@qd+?=?V@y?ǧ??R??"?
both?Your program is POTENTIALLY input-bound because 93.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?91.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 