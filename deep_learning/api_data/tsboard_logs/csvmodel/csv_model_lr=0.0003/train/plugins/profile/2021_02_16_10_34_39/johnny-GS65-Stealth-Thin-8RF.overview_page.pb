?	??4F?lL@??4F?lL@!??4F?lL@	???ɂ}????ɂ}?!???ɂ}?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??4F?lL@?S?*?J@1???E_??A9(a????I?ֈ`?@Y??????p?*	???Mb@R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??x@ٔ?!??[?;@)g?lt?O??1??=41(7@:Preprocessing2F
Iterator::ModelN?#~???!A?0x"D@)~?Ɍ????1*?Lǲt4@:Preprocessing2U
Iterator::Model::ParallelMapV2=~oӟ??!Y?o?=?3@)=~oӟ??1Y?o?=?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatenQf?L2??!B???(W8@)?pY?? ??1???#?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceNd???z?!?c??-?!@)Nd???z?1?c??-?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'???S??!??!χ?M@)[?kBZcp?1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorc?J!?Kl?!?u?1??@)c?J!?Kl?1?u?1??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$?P29???!祐=?\:@)i;???.X?1kMG?, @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???ɂ}?I???°?X@Q?? ?L???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S?*?J@?S?*?J@!?S?*?J@      ??!       "	???E_?????E_??!???E_??*      ??!       2	9(a????9(a????!9(a????:	?ֈ`?@?ֈ`?@!?ֈ`?@B      ??!       J	??????p???????p?!??????p?R      ??!       Z	??????p???????p?!??????p?b      ??!       JGPUY???ɂ}?b q???°?X@y?? ?L????"I
+gradient_tape/sequential_16/dense_49/MatMulMatMul1[???N??!1[???N??0"?
&gradient_tape/mean_squared_error/mul_1Mul?b!N??!?(	?_N??";
sequential_16/dense_48/MatMulMatMul0?Y??!?BKAp???0";
sequential_16/dense_49/MatMulMatMul0?Y??!?𑗰3??0"I
+gradient_tape/sequential_16/dense_48/MatMulMatMulT??s??!腅?͹??0"I
-gradient_tape/sequential_16/dense_49/MatMul_1MatMul??)???!?dW?????"I
-gradient_tape/sequential_16/dense_50/MatMul_1MatMul	??R??z?!vކ??:??";
sequential_16/dense_50/MatMulMatMul	??R??z?!?W?????0"Y
8gradient_tape/sequential_16/dense_50/BiasAdd/BiasAddGradBiasAddGradXgv?!??ܱ?D??"1
Nadam/Nadam/update_8/mulMulT??sv?!???#???Q      Y@Yu?E]t@a颋.??W@q?=Tv?4W@y	??됞??"?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?92.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 