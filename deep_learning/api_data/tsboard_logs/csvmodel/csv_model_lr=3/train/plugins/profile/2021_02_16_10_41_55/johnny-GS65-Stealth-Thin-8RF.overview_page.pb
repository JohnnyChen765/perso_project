?	D??]FN@D??]FN@!D??]FN@	r~s?ȵ?r~s?ȵ?!r~s?ȵ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6D??]FN@_y??"-L@1->?x???A</?:??IϠ?_	@YB???8a??*	8?A`??V@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???9}??!$??#?a?@)???E???1po???.:@:Preprocessing2U
Iterator::Model::ParallelMapV2a??>?̔?!jK??"6@)a??>?̔?1jK??"6@:Preprocessing2F
Iterator::Model??O?Y???!	?`?L;E@)O"¿??1???T4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?zM
J??!??cC=f2@)?*k??q??1?A̤?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??/?1"??!3???;"@)??/?1"??13???;"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn???V??!?	?d??L@)0?[w?t?1z~???K@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???͋s?!?f???@)???͋s?1?f???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?2???V??!+?٦x?4@)????fd`?1W???q@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9r~s?ȵ?I3@??X@QO?7A?a??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	_y??"-L@_y??"-L@!_y??"-L@      ??!       "	->?x???->?x???!->?x???*      ??!       2	</?:??</?:??!</?:??:	Ϡ?_	@Ϡ?_	@!Ϡ?_	@B      ??!       J	B???8a??B???8a??!B???8a??R      ??!       Z	B???8a??B???8a??!B???8a??b      ??!       JGPUYr~s?ȵ?b q3@??X@yO?7A?a???"I
+gradient_tape/sequential_24/dense_73/MatMulMatMul
??????!
??????0";
sequential_24/dense_73/MatMulMatMul.???S???!z?z????0"I
+gradient_tape/sequential_24/dense_72/MatMulMatMul?oʟc???!?XA%???0";
sequential_24/dense_72/MatMulMatMul?oʟc???!??3?ɧ?0"I
-gradient_tape/sequential_24/dense_73/MatMul_1MatMul~?;????!????`Q??"I
-gradient_tape/sequential_24/dense_74/MatMul_1MatMulv??Qf0{?!????m???";
sequential_24/dense_74/MatMulMatMulv??Qf0{?!?b?-????0"Y
8gradient_tape/sequential_24/dense_74/BiasAdd/BiasAddGradBiasAddGradW?????v?!??E???"I
+gradient_tape/sequential_24/dense_74/MatMulMatMulW?????v?!8}???c??0"$

Nadam/CastCast?oʟc?v?!2$?Dε?Q      Y@Yg<??x@a9?as?W@q??l???V@y???<Q??"?
both?Your program is POTENTIALLY input-bound because 93.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?91.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 