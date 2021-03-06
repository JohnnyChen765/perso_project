?	6??,
L@6??,
L@!6??,
L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-6??,
L@Q?E?3J@1"? ˂I??A80?Qd???I??e??4@*	?ZdKU@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??p?????!$9vu);@)?H????1"t(.??6@:Preprocessing2F
Iterator::Model?2??֡?!y??*?sD@)9F?G???1|Cix?4@:Preprocessing2U
Iterator::Model::ParallelMapV2TH?9???!?X?}'4@)TH?9???1?X?}'4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateK;5???!t??q!8@)???M?q??1;??SQ.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceUm7?7M?!??̏?!@)Um7?7M?1??̏?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?5?U?ũ?!?22??M@)>]ݱ?&u?1|a?@@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??l??po?!	7??@)??l??po?1	7??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8-x?W???!??U??9@)?/??CX?1s??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????}X@Q?????A @Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Q?E?3J@Q?E?3J@!Q?E?3J@      ??!       "	"? ˂I??"? ˂I??!"? ˂I??*      ??!       2	80?Qd???80?Qd???!80?Qd???:	??e??4@??e??4@!??e??4@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????}X@y?????A @?"I
+gradient_tape/sequential_23/dense_70/MatMulMatMulh5???ڇ?!h5???ڇ?0";
sequential_23/dense_69/MatMulMatMulh5???ڇ?!h5???ڗ?0"I
+gradient_tape/sequential_23/dense_69/MatMulMatMul?tQ????!???2?n??0";
sequential_23/dense_70/MatMulMatMul?tQ????!U????0"3
Nadam/Nadam/update_3/mul_2MulD  ??.??!Uf?????"I
-gradient_tape/sequential_23/dense_70/MatMul_1MatMulD  ??.??!?*W????"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchn]??&?y?!lpbs???"Y
8gradient_tape/sequential_23/dense_71/BiasAdd/BiasAddGradBiasAddGradn]??&?y?!C?mਹ??"I
-gradient_tape/sequential_23/dense_71/MatMul_1MatMuln]??&?y?!?xM?T??";
sequential_23/dense_71/MatMulMatMul??d("v?!?D??̶??0Q      Y@Yg<??x@a9?as?W@q??!?/AX@y"?/G?3??"?
both?Your program is POTENTIALLY input-bound because 93.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 