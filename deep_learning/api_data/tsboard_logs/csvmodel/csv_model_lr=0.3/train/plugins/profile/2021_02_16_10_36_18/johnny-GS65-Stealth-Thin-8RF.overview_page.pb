?	?V횐?J@?V횐?J@!?V횐?J@	̺??M??̺??M??!̺??M??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?V횐?J@@0G???H@1??$"????A_????@??I?q4GV?@YZe??????*	ˡE???S@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? ?b??!??????<@)/?
ҌE??1??$???7@:Preprocessing2U
Iterator::Model::ParallelMapV2?qs*??!ͻ?va?3@)?qs*??1ͻ?va?3@:Preprocessing2F
Iterator::Model??1????!K??yx?C@)???հ??1?܇{??3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?=~o??!?i?5??7@)o?????1K???H~-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?%8???}?!?F5??l"@)?%8???}?1?F5??l"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?_?|x???!?3???ON@)??ƽ?s?1?(?$W}@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHk:!tp?!?mP?H@)Hk:!tp?1?mP?H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq?i???!?,???9@)d??uY?1~)L?c??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9˺??M??Ic\??}?X@QP?????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@0G???H@@0G???H@!@0G???H@      ??!       "	??$"??????$"????!??$"????*      ??!       2	_????@??_????@??!_????@??:	?q4GV?@?q4GV?@!?q4GV?@B      ??!       J	Ze??????Ze??????!Ze??????R      ??!       Z	Ze??????Ze??????!Ze??????b      ??!       JGPUY˺??M??b qc\??}?X@yP??????"I
+gradient_tape/sequential_22/dense_67/MatMulMatMul??`zB???!??`zB???0";
sequential_22/dense_66/MatMulMatMulk?????!??:9?ŗ?0"I
+gradient_tape/sequential_22/dense_66/MatMulMatMul???-???!ߘZb????0";
sequential_22/dense_67/MatMulMatMulDs?u9`??!??̿????0"I
-gradient_tape/sequential_22/dense_67/MatMul_1MatMulZਔ?_??!?-?䃻??"I
-gradient_tape/sequential_22/dense_68/MatMul_1MatMul?Ջ6*{?!??h?? ??";
sequential_22/dense_68/MatMulMatMul?Ջ6*{?!???C??0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch???-?v?!Z\?;???"Y
8gradient_tape/sequential_22/dense_68/BiasAdd/BiasAddGradBiasAddGrad???-?v?!???Xn??"*
div_no_nan_1DivNoNan_??Er?!~??B9??Q      Y@Yu?E]t@a颋.??W@q??}?d?R@y	??E?8??"?
both?Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?74.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 