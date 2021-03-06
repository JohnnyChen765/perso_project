?	?????ML@?????ML@!?????ML@	?1O+a???1O+a??!?1O+a??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?????ML@?????J@1???҇.??A?eN??Ħ?Ih%???b@Y0?Qd????*	?t??R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX?????!Z??g?:@)bMeQ?E??1?V???l6@:Preprocessing2U
Iterator::Model::ParallelMapV2??Cl??!E)?]R5@)??Cl??1E)?]R5@:Preprocessing2F
Iterator::Model+?`??!??? dCE@)?|a2U??1I??jj45@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?.????!??^???8@)?0DN_χ?1?5??.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/3l??{?!?һ?P&"@)/3l??{?1?һ?P&"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??X?_"??!Z???L@)Oʤ?6 k?1?f`Y	?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??u??i?!ľD???@)??u??i?1ľD???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??H??_??!?z!Aes:@)???1??W?1??(l4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?1O+a??Ib%???X@Q{?%q???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????J@?????J@!?????J@      ??!       "	???҇.?????҇.??!???҇.??*      ??!       2	?eN??Ħ??eN??Ħ?!?eN??Ħ?:	h%???b@h%???b@!h%???b@B      ??!       J	0?Qd????0?Qd????!0?Qd????R      ??!       Z	0?Qd????0?Qd????!0?Qd????b      ??!       JGPUY?1O+a??b qb%???X@y{?%q????"I
+gradient_tape/sequential_18/dense_55/MatMulMatMulY?\?Yu??!Y?\?Yu??0"S
:sequential_18/batch_normalization_37/AssignMovingAvg_1/mulMulY?\?Yu??!Y?\?Yu??"I
+gradient_tape/sequential_18/dense_54/MatMulMatMul???2?<??!$=Q?ɡ?0";
sequential_18/dense_54/MatMulMatMul???2?<??!?5?cY??0";
sequential_18/dense_55/MatMulMatMul???2?<??!.??1???0"I
-gradient_tape/sequential_18/dense_55/MatMul_1MatMul5?z??!qz?(????"I
-gradient_tape/sequential_18/dense_56/MatMul_1MatMulZ?Ҧ? ?!G??????"Y
8gradient_tape/sequential_18/dense_56/BiasAdd/BiasAddGradBiasAddGrad8??=v?!X(??kJ??";
sequential_18/dense_56/MatMulMatMul8??=v?!i??????0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch????"<v?!??O!??Q      Y@Yu?E]t@a颋.??W@qY?v??$T@yZ??+???"?
both?Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?80.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 