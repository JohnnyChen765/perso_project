?	I?0elL@I?0elL@!I?0elL@	?A????A???!?A???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6I?0elL@???g%?J@1wj.7???A??????I??Y??@Yq?;??*	??? ??U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?a??h???!]??A??;@)?????.??1?F????7@:Preprocessing2F
Iterator::Model	?<??t??!?z?*?D@)+Kt?Y???1TO禗?4@:Preprocessing2U
Iterator::Model::ParallelMapV2?x?'e??!??K??4@)?x?'e??1??K??4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateyY|??!? ??D 8@)(???,??1 ????=-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	?^)ˀ?!tk'??"@)	?^)ˀ?1tk'??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip̴?++M??!???aM@)??ᱟ?r?1??N??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?{?ԗ?m?!??I?@)?{?ԗ?m?1??I?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapߣ?z???!7jb
ַ9@)u?Rz??X?1ܖvy??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?A???I??1??X@Q	ξ??O??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???g%?J@???g%?J@!???g%?J@      ??!       "	wj.7???wj.7???!wj.7???*      ??!       2	????????????!??????:	??Y??@??Y??@!??Y??@B      ??!       J	q?;??q?;??!q?;??R      ??!       Z	q?;??q?;??!q?;??b      ??!       JGPUY?A???b q??1??X@y	ξ??O???"I
+gradient_tape/sequential_15/dense_46/MatMulMatMul`?%?M???!`?%?M???0"I
+gradient_tape/sequential_15/dense_45/MatMulMatMul?N? o??!
y?`'???0";
sequential_15/dense_45/MatMulMatMul?N? o??!2	??b??0";
sequential_15/dense_46/MatMulMatMul?N? o??!???'????0"I
-gradient_tape/sequential_15/dense_46/MatMul_1MatMul?#?C1??!?l??
??"3
Nadam/Nadam/update_4/mul_6Mul]?2%g???!l?C??"Y
8gradient_tape/sequential_15/dense_47/BiasAdd/BiasAddGradBiasAddGrad6K>޹?z?!?? 
????"I
-gradient_tape/sequential_15/dense_47/MatMul_1MatMul6K>޹?z?!u???V???"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch?N? ov?!`?̶F??"e
>sequential_15/batch_normalization_30/moments/SquaredDifferenceSquaredDifference?N? ov?!K???6o??Q      Y@Yu?E]t@a颋.??W@qY???? S@ya"*?
??"?
both?Your program is POTENTIALLY input-bound because 93.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?76.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 