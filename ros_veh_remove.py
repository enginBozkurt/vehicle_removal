#!/usr/bin/python

'''
Aim: This code attempts a Multi-Obejct Tracking based vehicle removal approach
using PointRCNN (trained to detect all four-wheel vehicles) as backend dectector
and an MHEKF as the backend tracker.

Author: Dr. Saurab Verma
'''

if True:
	import rospy
	import torch
	import logging
	import numpy as np
	from time import time
	from tqdm import tqdm
	from rosbag import Bag
	from pathlib import Path
	from std_msgs.msg import String
	from collections import namedtuple
	from squaternion import quat2euler
	from sensor_msgs.msg import PointField
	import _init_path
	from lib.net.point_rcnn import PointRCNN
	from lib.utils.iou3d.iou3d_utils import nms_gpu
	from lib.utils.kitti_utils import boxes3d_to_bev_torch
	from lib.utils.bbox_transform import decode_bbox_target
	from tools.train_utils.train_utils import load_checkpoint
	from lib.config import cfg, cfg_from_file, save_config_to_file


class lidar_clean:

	def __init__(self, node_name=''):

		self.rosparam_(node_name)

		self.create_logger_(node_name)
		self.logger.info('ROS parameters loaded successfully')
		self.logger.info(
			'********************** Start logging **********************')
		save_config_to_file(cfg, logger=self.logger)
		self.logger.info('ROS buffer_len: {}'.format(self.buffer_len))
		self.logger.info('ROS lidar_topic: {}'.format(self.lidar_topic))
		self.logger.info('ROS debug_flag: {}'.format(self.debug_flag))
		self.logger.info(
			'ROS depth_threshold: {}'.format(self.depth_threshold))
		self.logger.info(
			'ROS score_threshold: {}'.format(self.score_threshold))
		self.logger.info('ROS model_checkpoint: {}'.format(self.model_checkpoint))

		self.model = PointRCNN(num_classes=self.num_class,
							   use_xyz=True, mode='TEST')
		self.model.cuda()
		load_checkpoint(
			self.model,
			filename=str(self.base_dir / self.model_checkpoint),
			logger=self.logger)
		self.model.eval()

		self.logger.info('Model initialization complete')

	def rosbag_handler(self):
		'''
		Input the name of the rosbag to be processed by the model.
		Outputs boolean value indicating success.
		Meanwhile also stores new rosbag with 'cleaned' lidar data.
		'''

		if not (self.base_dir / self.input_bags_path).is_dir():
			self.logger.error('No input dir found: {}'.format(
				self.base_dir / self.input_bags_path))
			return False

		input_bags = sorted(
			(self.base_dir / self.input_bags_path).glob('*.bag'))
		if not input_bags:
			self.logger.error('No input rosbag found at dir: {}'.format(
				self.base_dir / self.input_bags_path))
			return False
		else:
			print('Number of input rosbags: {}'.format(len(input_bags)))

		input_bags_filtered = []
		self.output_bags = []
		for input_bag in input_bags:
			if not input_bag.is_file():
				self.logger.error(
					'input rosbag not available: {}'.format(input_bag))
			else:
				input_bags_filtered.append(input_bag)
				self.output_bags.append(
					self.output_dir / (self.output_bag_prefix + input_bag.name[:-4] + '.bag'))

		if not input_bags_filtered:
			self.logger.error('No files detected as input rosbag at dir: {}'.format(
				self.base_dir / self.input_bags_path))
		else:
			input_bags = input_bags_filtered

		counter = 1
		self.nt = namedtuple(
			'Tracking_buffer',
			'topic ' +
			'ros_time ' +
			'lidar_scan_msg ' +
			'prev_tracked_vehicles ' +
			'filtered_bboxes')
		self.tracking_buffer_list = []
		self.enc_l_val, self.enc_r_val, self.imu_msg_list = 0.0, 0.0, [0.0]*4

		print('Processing bag files . . .')
		start_time = time()
		pbar_bags = tqdm(total=len(input_bags), position=0, desc='Rosbag: ')
		for input_bag in input_bags:
			with Bag(input_bag) as inbag:

				self.outbag = Bag(self.output_bags[0], 'w')
				self.logger.info(
					'Opened new output bag: {}'.format(self.output_bags[0]))
				self.output_bags.pop(0)

				self.set_outbag_()

				pbar_bags.set_postfix(dict(inbag=input_bag.name[:-4]))
				self.logger.info(
					'{}: Processing bag file: {}'.format(counter, input_bag))
				counter += 1

				if not self.process_bags_(inbag):
					self.outbag_lidar.close()
					return False

				self.input_changed_flag = True
				pbar_bags.update()
		pbar_bags.close()

		for item in self.tracking_buffer_list:

			t = item.ros_time
			topic = item.topic
			msg = item.lidar_scan_msg
			bboxes = item.filtered_bboxes

			msg, label_msg = self.clean_pointcloud_(t.to_sec(), msg, bboxes)

			self.outbag_lidar.write(topic, msg, t)
			self.outbag_lidar.write(self.label_topic, label_msg, t)
		self.outbag_lidar.close()

		print()
		self.logger.info('Process completed successfully')
		self.logger.info('Overall time taken for processing: {}'.format(time()-start_time))
		return True

	def set_outbag_(self):

		try:
			self.outbag_lidar
		except AttributeError:
			self.outbag_lidar = None
			self.inbag_msg_cnt_list = []

		if self.outbag_lidar:
			if self.outbag_msg_count == self.inbag_msg_cnt_list[0] and self.input_changed_flag:
				self.input_changed_flag = False
				self.outbag_msg_count = 0
				self.inbag_msg_cnt_list.pop(0)
				self.outbag_lidar.close()
				self.outbag_lidar = self.outbag
				self.logger.info(
					'Lidar output rosbag set to same as the other open output rosbag')
		else:
			self.input_changed_flag = False
			self.outbag_msg_count = 0
			self.outbag_lidar = self.outbag
			self.logger.info(
				'Lidar output rosbag set to same as the other open output rosbag')

	def process_bags_(self, input_bag):

		return_flag = True

		msg_counter = 0
		inbag_msg_cnt = input_bag.get_message_count(self.lidar_topic)
		self.inbag_msg_cnt_list.append(inbag_msg_cnt)
		pbar = tqdm(total=inbag_msg_cnt, position=1, desc='Lidar msg: ')

		for topic, msg, t in input_bag.read_messages():

			if self.debug_flag and (msg_counter > self.debug_msg_limit):
				self.logger.info(
					'ROS node debugging finished sucessfully for current input rosbag!')
				break

			if rospy.is_shutdown():
				return_flag = False
				print()
				self.logger.error('ROS node is shutdown!')
				break

			if topic in self.imu_topic:
				self.imu_msg_list = [msg.orientation.w, msg.orientation.x,
									 msg.orientation.y, msg.orientation.z]

			if topic in self.left_wheel_enc_topic:
				self.enc_l_val = msg.data

			if topic in self.right_wheel_enc_topic:
				self.enc_r_val = msg.data

			if topic in self.lidar_topic:

				msg_counter += 1

				if msg.height is not 1:
					return_flag = False
					print()
					self.logger.error(
						'height={} for msg at time {}'.format(msg.height, t.to_sec()))
					break

				all_detected_bboxes = self.detect_vehicles_(msg, t.to_sec())

				self.enhanced_mot_(
					topic, t, msg, all_detected_bboxes, self.odom_process_())

				pbar.update()

				if len(self.tracking_buffer_list) > self.buffer_len:

					item = self.tracking_buffer_list.pop(0)
					t = item.ros_time
					topic = item.topic
					msg = item.lidar_scan_msg
					bboxes = item.filtered_bboxes

					msg, label_msg = self.clean_pointcloud_(
						t.to_sec(), msg, bboxes)

					self.outbag_lidar.write(topic, msg, t)
					self.outbag_lidar.write(self.label_topic, label_msg, t)
					self.outbag_msg_count += 1
					self.set_outbag_()
			else:
				self.outbag.write(topic, msg, t)

		pbar.close()

		return return_flag

	def odom_process_(self):

		l_enc, r_enc, imu = self.enc_l_val, self.enc_r_val, self.imu_msg_list

		_, _, current_imu_yaw = quat2euler(imu[0], imu[1], imu[2], imu[3])

		if self.first_time_odom_flag:
			self.first_time_odom_flag = False
			self.imu_yaw_prev = current_imu_yaw
			self.l_enc_prev = l_enc
			self.r_enc_prev = r_enc
			return np.zeros(3)

		if np.isnan(current_imu_yaw):
			self.logger.error(
				'current_imu_yaw value error: {}'.format(current_imu_yaw))

		delta_q = current_imu_yaw - self.imu_yaw_prev

		delta_l_enc = l_enc - self.l_enc_prev
		delta_r_enc = self.r_enc_prev - r_enc

		if (abs(delta_l_enc) > self.filter_threshold):
			self.logger.error("delta_l_enc jump: {}".format(delta_l_enc))
			delta_l_enc = 0
			delta_r_enc = 0

		if (abs(delta_r_enc) > self.filter_threshold):
			self.logger.error("delta_r_enc jump: {}".format(delta_r_enc))
			delta_l_enc = 0
			delta_r_enc = 0

		l_wd = delta_l_enc * self.tick_distance_
		r_wd = delta_r_enc * self.tick_distance_

		delta_v = (l_wd + r_wd) / 2.

		self.imu_yaw_prev = current_imu_yaw
		self.l_enc_prev = l_enc
		self.r_enc_prev = r_enc

		self.enc_l_val, self.enc_r_val, self.imu_msg_list = 0.0, 0.0, [0.0]*4

		return np.array([delta_v*np.cos(delta_q), delta_v*np.sin(delta_q), delta_q])

	def detect_vehicles_(self, msg, t):

		if not self.np_dtype_list:
			self.pointcloud2_to_dtype_(msg)

		pts_input_dict = np.fromstring(msg.data, self.np_dtype_list)
		pts_input = np.array([pts_input_dict[:]['x'],
							  pts_input_dict[:]['y'],
							  pts_input_dict[:]['z']]).T

		if self.debug_flag:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(pts_input)
			o3d.io.write_point_cloud(
				'{}/{}.pcd'.format(self.pcd_input_dir, t), pcd)

		pts_depth = pts_input[:, 0]
		pts_front_flag = pts_depth > 0.0
		choice = np.where(pts_front_flag)[0]
		pts_input = pts_input[choice, :]

		len_pts_input = len(pts_input)
		if self.npoints < len_pts_input:
			pts_near_flag = pts_depth < self.depth_threshold
			far_idxs_choice = np.where(
				np.logical_not(pts_near_flag))[0]
			near_idxs = np.where(pts_near_flag)[0]
			near_idxs_choice = np.random.choice(
				near_idxs, self.npoints - len(far_idxs_choice),
				replace=(len(near_idxs) < self.npoints - len(far_idxs_choice)))
			choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
				if len(far_idxs_choice) > 0 else near_idxs_choice
		else:
			self.logger.debug(
				'{} extra points added, as model input, for scan at time: {}'.format(
					self.npoints - len_pts_input, t))
			choice = np.arange(
				0, len_pts_input, dtype=np.int32)
			extra_choice = np.random.choice(
				choice, self.npoints - len_pts_input,
				replace=(len_pts_input < self.npoints - len_pts_input))
			choice = np.concatenate(
				(choice, extra_choice), axis=0)
		np.random.shuffle(choice)
		pts_input = pts_input[choice, :]

		return self.forward_pass(pts_input)

	def clean_pointcloud_(self, t, msg, bboxes):

		pts_dict = np.fromstring(msg.data, self.np_dtype_list)
		pts_output = np.array([pts_dict[:]['x'],
							   pts_dict[:]['y'],
							   pts_dict[:]['z']]).T

		pts_depth = pts_output[:, 0]
		pts_front_flag = pts_depth > 0.0
		choice = np.where(pts_front_flag)[0]
		pts_output = pts_output[choice, :]
		pts_dict = pts_dict[choice]

		pts_front_refined = np.arctan(pts_output[:, 1]/pts_output[:, 0])
		pts_front_refined_flag = (-np.pi/4. <= pts_front_refined) * \
			(pts_front_refined <= np.pi/4.)
		choice = np.where(pts_front_refined_flag)[0]
		pts_output = pts_output[choice, :]
		pts_dict = pts_dict[choice]

		original_pts_dict_len = len(pts_dict)

		for bb in bboxes:

			pts_z = pts_output[:, 2]

			pos_x = bb[0]
			pos_y = bb[1]
			pos_z = bb[2]
			height = bb[3]
			width = bb[4]
			length = bb[5]
			yaw = bb[6]

			cy = np.cos(yaw)
			sy = np.sin(yaw)
			A = np.array([[cy, -cy, -sy, sy],
			B = np.array([
				[0.5, 0, cy, sy],
				[0.5, 0, -cy, -sy],
				[0, 0.5, -sy, cy],
				[0, 0.5, sy, -cy]
			B = np.matmul(B, np.array(

			pts_inside_flag = (
			pts_inside_flag = np.logical_and.reduce(
			pts_inside_flag = (pts_z > pos_z) & \
				(pts_z < pos_z + height) & \
			outside_pts_ind = np.where(
			if (not np.any(pts_inside_flag)):
				self.logger.warn(
					'Expected but no points removed from scan at time {}'.format(t))
			pts_dict = pts_dict[outside_pts_ind]

		if original_pts_dict_len != pts_dict.size:
			self.logger.info('{} points removed from scan at time {}'.format(
				msg.width-pts_dict.size, t))

		msg.width = len(pts_dict)
		msg.row_step = msg.width * msg.point_step
		msg.data = pts_dict.tobytes()

		format_string = ''
		template_string = 'Car -1 -1 {:.2f} 0.0 0.0 0.0 0.0' + \
			'{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'
		for bb in bboxes:
			beta = np.arctan2(bb[2], bb[0])
			alpha = -np.sign(beta) * \
				np.pi / 2 + beta + bb[6]
			format_string += template_string.format(
				alpha, bb[3], bb[4], bb[5], bb[0], bb[1], bb[2], bb[6], bb[7])
		format_string = format_string[:-1]

		if self.debug_flag:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(pts_output)
			o3d.io.write_point_cloud(
				'{}/{}.pcd'.format(self.pcd_output_dir, t), pcd)

			with open('{}/{}.txt'.format(self.label_output_dir, t), 'w') as f:
				f.write(format_string)

		label_msg = String()
		label_msg.data = 'header_time: {}\n'.format(
			msg.header.stamp.to_sec()) + format_string

		return msg, label_msg

	def enhanced_mot_(self, topic, t, msg, detected_bboxes, odom):
		'''
		Apply Multi-Hypothesis Extended Kalman Filter to detected observations

		Input: Detected bounding boxes information [Type: list of list, the
		latter being specifically {pos_x, pos_y, pos_z, height, width, length,
		yaw, score}; described in lidar frame]
		Input: Odometer info [Type: list of int i.e. {vel_x, vel_y, vel_yaw}]
		Output: Filtered bounding boxes information based on tracking
		'''


		new_tracked_vehicles = []
		filtered_bboxes_list = []
		prev_tracked_vehicles = []

		if len(self.vehicles_being_tracked) > 10:
			self.logger.warn('Something is mistaken, tracking vehicles are too many: {}'.format(
				len(self.vehicles_being_tracked)))


		for ID, states, weights, other_state_info in self.vehicles_being_tracked:

			if detected_bboxes.shape[0] == 0:
				self.logger.warn(
					'Not enough vehicles detected, as are tracked')
				break

			pos_x_array = detected_bboxes[:, 0]
			pos_y_array = detected_bboxes[:, 1]
			yaw_array = detected_bboxes[:, 6]

			states[:, 0] += states[:, 3] * \
				np.cos(states[:, 2]) * self.time_period
			states[:, 1] += states[:, 3] * \
				np.sin(states[:, 2]) * self.time_period
			states[:, 2] += states[:, 3] * states[:, 4] * self.time_period
			states[:, 3] += np.random.normal(0.0,
											 self.linear_speed_sigma,
											 self.hypothesis_len)
			states[:, 4] += np.random.normal(0.0,
											 self.curvature_sigma,
											 self.hypothesis_len)

			states[:, 0] -= odom[0]
			states[:, 1] -= odom[1]
			states[:, 2] -= odom[2]

			hypothesis_index = np.zeros(weights.size, dtype=np.uint64)
			old_weights = weights.copy()
			for i in range(self.hypothesis_len):
				z = np.array([
					pos_x_array - states[i, 0],
					pos_y_array - states[i, 1],

				lambda_value = np.exp(-0.5 *
									  np.diag(
										  np.matmul(
											  z.T, np.matmul(
												  self.Z_inv, z)

				hypothesis_index[i] = np.argmax(lambda_value)
				weights[i] *= np.max([self.lambda_min, np.max(lambda_value)])

			if np.sum(weights) > 1.1*self.lambda_min:

				weights /= np.sum(weights)

				bb_index_used = hypothesis_index[np.argmax(weights)]

				new_tracked_vehicles.append((
					ID,
					states,
					weights,
				))

				filtered_bboxes_list.append(detected_bboxes[bb_index_used])
				detected_bboxes = np.delete(detected_bboxes, bb_index_used, 0)
			else:

				self.logger.warn(
					'Vehicle ID: {} is now NOT tracked anymore at time {}'.format(ID, t.to_sec()))

				prev_tracked_vehicles.append((
					ID,
					states,
					old_weights,
					other_state_info,
				))



		if self.tracking_buffer_list:

			ID_revived = []
			for ID, states, weights, other_state_info, track_num in self.tracking_buffer_list[-1].prev_tracked_vehicles:

				if detected_bboxes.shape[0] == 0:
					self.logger.warn(
						'Not enough vehicles detected, as were previously tracked')
					break

				pos_x_array = detected_bboxes[:, 0]
				pos_y_array = detected_bboxes[:, 1]
				yaw_array = detected_bboxes[:, 6]

				states[:, 0] += states[:, 3] * \
					np.cos(states[:, 2]) * self.time_period
				states[:, 1] += states[:, 3] * \
					np.sin(states[:, 2]) * self.time_period
				states[:, 2] += states[:, 3] * states[:, 4] * self.time_period
				states[:, 3] += np.random.normal(0.0,
												 self.linear_speed_sigma,
												 self.hypothesis_len)
				states[:, 4] += np.random.normal(0.0,
												 self.curvature_sigma,
												 self.hypothesis_len)

				states[:, 0] -= odom[0]
				states[:, 1] -= odom[1]
				states[:, 2] -= odom[2]

				hypothesis_index = np.zeros(weights.size, dtype=np.uint64)
				old_weights = weights.copy()
				for i in range(self.hypothesis_len):
					z = np.array([
						pos_x_array - states[i, 0],
						pos_y_array - states[i, 1],

					lambda_value = np.exp(-0.5 *
										  np.diag(
											  np.matmul(
												  z.T, np.matmul(
													  self.Z_inv, z)

					hypothesis_index[i] = np.argmax(lambda_value)
					weights[i] *= np.max([self.lambda_min,
										  np.max(lambda_value)])

				if np.sum(weights) > 1.1*self.lambda_min:

					self.logger.warn(
						'Vehicle ID: {} has resumed tracking again at time {}'.format(ID, t.to_sec()))

					ID_revived.append(ID)

					weights /= np.sum(weights)

					bb_index_used = hypothesis_index[np.argmax(weights)]

					new_tracked_vehicles.append((
						ID,
						states,
						weights,
					))

					filtered_bboxes_list.append(detected_bboxes[bb_index_used])
					detected_bboxes = np.delete(
						detected_bboxes, bb_index_used, 0)
				else:
					if track_num <= self.buffer_len:
						prev_tracked_vehicles.append((
							ID,
							states,
							old_weights,
							other_state_info,
						))

			if ID_revived:
				for ind in range(len(self.tracking_buffer_list)-1):

					item = self.tracking_buffer_list[ind]
					filtered_bboxes = item.filtered_bboxes

					filtered_prev_tracked_vehicles = []
					for ID, states, weights, other_state_info, track_num in item.prev_tracked_vehicles:

						if ID in ID_revived:
							bbox[0, 0] = np.sum(weights * states[:, 0].T)
							bbox[0, 1] = np.sum(weights * states[:, 1].T)
							bbox[0, 2:6] = other_state_info
							bbox[0, 6] = np.sum(weights * states[:, 2].T)


							filtered_bboxes = np.append(
								filtered_bboxes, bbox, axis=0)

						else:
							filtered_prev_tracked_vehicles.append((
								ID, states, weights, other_state_info, track_num))

					self.tracking_buffer_list[ind] = self.nt(topic=item.topic,
															ros_time=item.ros_time,
															filtered_bboxes=filtered_bboxes,
															lidar_scan_msg=item.lidar_scan_msg,
															prev_tracked_vehicles=filtered_prev_tracked_vehicles
															)


		for bb in detected_bboxes:
			if bb[7] > self.score_threshold:

				filtered_bboxes_list.append(bb)

				pos_x = bb[0]
				pos_y = bb[1]
				yaw = bb[6]
				self.ID_used += 1
				new_tracked_vehicles.append((
					self.ID_used,
					np.array([[pos_x, pos_y, yaw, ls, cur]
							  for ls, cur in self.hypothesis]),
					np.array([1./self.hypothesis_len]*self.hypothesis_len),
					bb[2:6]
				))
				self.logger.info(
					'Vehicle ID: {} is now being tracked starting time {}'.format(self.ID_used, t.to_sec()))


		self.vehicles_being_tracked = new_tracked_vehicles

		if filtered_bboxes_list:
			filtered_bboxes = np.array(filtered_bboxes_list)
		else:
			filtered_bboxes = np.zeros([0, 8])

		buffer_data = self.nt(topic=topic,
							  ros_time=t,
							  lidar_scan_msg=msg,
							  filtered_bboxes=filtered_bboxes,
							  prev_tracked_vehicles=prev_tracked_vehicles
							  )
		self.tracking_buffer_list.append(buffer_data)

	def forward_pass(self, pts_input):
		'''
		A forward pass on the NN model

		Input: raw lidar points [Type: 2D numpy array of size (self.npoints, 3)]
		Output: bounding boxes information [Type: list of list, the latter being
		specifically {pos_x, pos_y, pos_z, height, width, length, yaw,
		score}; described in lidar frame]
		'''

		pts_input = np.concatenate(
			(pts_input, np.ones([pts_input.shape[0], 1])), axis=1)
		pts_input = np.matmul(self.tf_lidar_wrt_camera, pts_input.T)
		pts_input = pts_input.T[:, :3]

		pts_input = torch.from_numpy(pts_input).cuda(

		nn_output = self.model(nn_input)

		rcnn_reg = nn_output['rcnn_reg'].view(
		scores = nn_output['rcnn_cls'].view(
			1, -1, nn_output['rcnn_cls'].shape[1])

		pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7),
										  rcnn_reg.view(-1,
														rcnn_reg.shape[-1]),
										  anchor_size=torch.from_numpy(
			cfg.CLS_MEAN_SIZE[0]).cuda(),
			loc_scope=cfg.RCNN.LOC_SCOPE,
			loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
			num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
			get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
			loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
			loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
			get_ry_fine=True
		).view(1, -1, 7)[0]
		scores = scores[0]

		boxes_bev_selected = boxes3d_to_bev_torch(pred_boxes3d)
		keep_idx = nms_gpu(boxes_bev_selected, scores,
						   cfg.RCNN.NMS_THRESH).view(-1)
		pred_boxes3d = pred_boxes3d[keep_idx]
		scores = scores[keep_idx]

		pred_boxes3d = pred_boxes3d.cpu().numpy()
		scores = scores.cpu().numpy()

		pts_temp = np.ones([pred_boxes3d.shape[0], 4])
		pts_temp[:, :3] = pred_boxes3d[:, :3]
		pts_output = np.matmul(self.tf_camera_wrt_lidar, pts_temp.T)
		pred_boxes3d[:, :3] = pts_output.T[:, :3]
		pred_boxes3d[:, 6] = -(np.pi/2.0 + pred_boxes3d[:, 6])


		return np.concatenate((pred_boxes3d, scores), axis=1)

	def rosparam_(self, node_name):

		self.buffer_len = int(rospy.get_param('buffer_len', 20))
		self.score_threshold = rospy.get_param('score_threshold', 3.0)
		self.lidar_topic = rospy.get_param(
			'lidar_topic', '/velodyne_front/velodyne_points')
		self.model_checkpoint = Path(rospy.get_param(
			'model_checkpoint', 'rpn_rcnn_checkpoint.pth'))

		cfg_file = rospy.get_param('cfg_file', '../tools/cfgs/default.yaml')
		cfg_from_file(cfg_file)

		self.tf_lidar_wrt_camera = np.array(
			[[0, -1, 0, 0],
			 [0, 0, -1, -0.08],
			 [1, 0, 0, -0.27],
			 [0, 0, 0, 1]]
		)
		self.tf_camera_wrt_lidar = self.tf_lidar_wrt_camera.copy()
		self.tf_camera_wrt_lidar[:3, :3] = self.tf_lidar_wrt_camera[:3, :3].T
		self.tf_camera_wrt_lidar[3, :3] = - np. matmul(
			self.tf_lidar_wrt_camera[:3, :3].T,
			self.tf_lidar_wrt_camera[3, :3])

		self.vehicles_being_tracked = []
		self.ID_used = -1
		try:
			self.Z_inv = np.linalg.inv(Z)
		except np.linalg.LinAlgError:
			print('Covariance matrix is not invertible!')
			exit(1)
		self.lambda_min = np.exp(-0.5 * 10)

		linear_speed_period = rospy.get_param(
		curvature_min = rospy.get_param(
		curvature_period = rospy.get_param(
		linear_speed_hypothesis = np.arange(
			linear_speed_min, linear_speed_max, linear_speed_period)
		curvature_hypothesis = np.arange(
			curvature_min, curvature_max, curvature_period)
		self.hypothesis = [
			(l, r) for l in linear_speed_hypothesis for r in curvature_hypothesis]
		self.hypothesis_len = len(self.hypothesis)

		self.filter_threshold = 15000
		self.first_time_odom_flag = True
		self.imu_yaw_prev = 0.0
		self.l_enc_prev = 0
		self.r_enc_prev = 0

		self.np_dtype_list = []
		self.imu_topic = rospy.get_param('imu_topic', '/an_device/Imu')
		self.left_wheel_enc_topic = rospy.get_param(
			'left_wheel_enc_topic', '/encoder_left')
		self.right_wheel_enc_topic = rospy.get_param(
			'right_wheel_enc_topic', '/encoder_right')
		self.label_topic = rospy.get_param(
			'label_topic', '/bounding_boxes')
		self.log_topic = rospy.get_param('log_topic', '')
		self.debug_flag = rospy.get_param('debug_flag', False)
		self.debug_msg_limit = rospy.get_param('debug_msg_limit', 10**10)
		self.depth_threshold = rospy.get_param('depth_threshold', 40.0)
		self.npoints = rospy.get_param('npoints', 2**14)
		self.output_bag_prefix = rospy.get_param('output_bag_prefix', 'clean_')

		self.base_dir = Path(rospy.get_param('base_dir', '.'))
		self.rel_output_dir = Path(
			rospy.get_param('rel_output_dir', 'output_mot_v2'))
		self.input_bags_path = Path(
			rospy.get_param('input_bags_path', 'input'))
		self.log_file = Path(rospy.get_param('log_file', 'output.log'))

		self.output_dir = self.base_dir / self.rel_output_dir
		self.output_dir.mkdir(parents=True, exist_ok=True)

		if self.debug_flag:

			print('\n*** NOTE: Debug mode is ON ***\n')
			print('\t>> pcds and labels will be stored as separate files in dir: {}'.format(
				self.output_dir))
			print('\t>> maximum of messages to be processed are limited to: {}'.format(
				self.debug_msg_limit))
			print()

			self.pcd_input_dir = self.output_dir / 'pcds/input/'
			self.pcd_input_dir.mkdir(parents=True, exist_ok=True)
			self.pcd_output_dir = self.output_dir / 'pcds/output/'
			self.pcd_output_dir.mkdir(parents=True, exist_ok=True)
			self.label_output_dir = self.output_dir / 'labels/'
			self.label_output_dir.mkdir(parents=True, exist_ok=True)

	def create_logger_(self, node_name):
		self.logger = logging.getLogger(node_name)
		self.logger.setLevel(logging.DEBUG)

		fh = logging.FileHandler(self.output_dir / self.log_file)
		fh.setLevel(logging.DEBUG)

		ch = logging.StreamHandler()
		ch.setLevel(logging.ERROR)

		formatter = logging.Formatter(
			'%(asctime)s  %(levelname)5s  %(message)s')
		fh.setFormatter(formatter)
		ch.setFormatter(formatter)

		self.logger.addHandler(fh)
		self.logger.addHandler(ch)

	def pointcloud2_to_dtype_(self, cloud_msg):
		'''Convert a list of PointFields to a numpy record datatype.
		'''

		DUMMY_FIELD_PREFIX = '__'

		type_mappings = [(PointField.INT8, np.dtype('int8')),
						 (PointField.UINT8, np.dtype('uint8')),
						 (PointField.INT16, np.dtype('int16')),
						 (PointField.UINT16, np.dtype('uint16')),
						 (PointField.INT32, np.dtype('int32')),
						 (PointField.UINT32, np.dtype('uint32')),
						 (PointField.FLOAT32, np.dtype('float32')),
						 (PointField.FLOAT64, np.dtype('float64'))]
		pftype_to_nptype = dict(type_mappings)

		pftype_sizes = {PointField.INT8: 1,
						PointField.UINT8: 1,
						PointField.INT16: 2,
						PointField.UINT16: 2,
						PointField.INT32: 4,
						PointField.UINT32: 4,
						PointField.FLOAT32: 4,
						PointField.FLOAT64: 8}

		offset = 0
		self.np_dtype_list = []
		for f in cloud_msg.fields:
			while offset < f.offset:
				self.np_dtype_list.append(
					('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
				offset += 1
			self.np_dtype_list.append((f.name, pftype_to_nptype[f.datatype]))
			offset += pftype_sizes[f.datatype]

		while offset < cloud_msg.point_step:
			self.np_dtype_list.append(
				('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
			offset += 1


if __name__ == '__main__':
	node_name = 'lidar_clean'
	rospy.init_node(node_name)

	with torch.no_grad():
		obj = lidar_clean(node_name)
		if obj.rosbag_handler():
			print('Process completed successfully')
		else:
			print('Process failed, please see log info for more details')
