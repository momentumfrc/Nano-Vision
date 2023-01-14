package com.momentum4999.vision;

import edu.wpi.first.apriltag.AprilTag;
import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import edu.wpi.first.apriltag.AprilTagFieldLayout;
import edu.wpi.first.apriltag.AprilTagFields;
import edu.wpi.first.apriltag.AprilTagPoseEstimate;
import edu.wpi.first.apriltag.AprilTagPoseEstimator;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Quaternion;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation3d;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.Closeable;
import java.io.IOException;
import java.util.Optional;
import java.util.Properties;

public class AprilTagVision implements Closeable {
    public static final AprilTagFieldLayout FIELD;
    public static final double FIELD_WIDTH = 16.4846;
    public static final double FIELD_HEIGHT = 7.9248;

    private final Mat grayscaleBuffer = new Mat();
    private final AprilTagDetector detector = new AprilTagDetector();

    private final VisionNetworkTable table;
    private final AprilTagPoseEstimator poseEstimator;
    private final boolean annotate;

    private double lastX;
    private double lastY;
    private double lastYaw;

    public AprilTagVision(VisionNetworkTable table, Properties settings) {
        this.table = table;
        this.poseEstimator = new AprilTagPoseEstimator(estimatorConfig(settings));
        this.detector.setConfig(detectorConfig(settings));
        this.detector.addFamily(settings.getProperty("aprtag.family", "tag16h5"), 0);
        this.annotate = Boolean.parseBoolean(settings.getProperty("aprtag.annotate_video", "true"));
    }

    public void updateVideo(Mat colorBuffer) {
        if (!colorBuffer.empty()) {
            Imgproc.cvtColor(colorBuffer, grayscaleBuffer, Imgproc.COLOR_RGB2GRAY);

            int entries = 0;
            Translation3d pos = new Translation3d();
            double yaw = 0;

            for (AprilTagDetection tag : this.detector.detect(grayscaleBuffer)) {
                Transform3d detection = this.poseEstimator.estimate(tag);
                Optional<Pose3d> expectation = FIELD.getTagPose(tag.getId());
                if (expectation.isPresent()) {
                     /* Detection's Axes:   Correct Axes:
                             Z                Z
                            /                 | [tag]
                           . -- X             . -- Y
                           | [tag]           /
                           Y                X               */
                    detection = new Transform3d( // Remap the detected tag transform to use the correct (field) axes
                            new Translation3d(-detection.getZ(), detection.getX(), -detection.getY()),
                            new Rotation3d(-detection.getRotation().getZ(), detection.getRotation().getX(), -detection.getRotation().getY()));

                    Pose3d camPose = expectation.get().transformBy(detection.inverse());
                    pos = pos.plus(camPose.getTranslation());
                    yaw += Math.PI + camPose.getRotation().getZ();
                    entries++;
                }

                if (this.annotate) {
                    annotateTag(tag, colorBuffer);
                }
            }

            if (entries > 0) {
                // Average position and rotation between all tag detections
                pos = pos.div(entries);
                yaw = yaw / entries;

                if (this.annotate) {
                    annotateCoords(pos.getX(), pos.getY(), pos.getZ(), yaw, colorBuffer);
                }
                this.table.updatePose(pos.getX(), pos.getY(), pos.getZ(), yaw);
                this.lastX = pos.getX();
                this.lastY = pos.getY();
                this.lastYaw = yaw;
            }
        }
    }

    public void updateField(Mat fieldBuffer) {
        if (!this.annotate) {
            return;
        }

        for (AprilTag tag : FIELD.getTags()) {
            annotatePose(tag.pose.getX(), tag.pose.getY(), tag.pose.getRotation().getZ(), 255, 0, 255, 2, 10, Integer.toString(tag.ID), fieldBuffer);
        }
        annotatePose(this.lastX, this.lastY, this.lastYaw, 0, 255, 0, 2, 16, null, fieldBuffer);
    }

    private static AprilTagDetector.Config detectorConfig(Properties settings) {
        AprilTagDetector.Config cfg = new AprilTagDetector.Config();
        cfg.numThreads = Integer.parseInt(settings.getProperty("aprtag.threads", Integer.toString(cfg.numThreads)));
        cfg.quadDecimate = Float.parseFloat(settings.getProperty("aprtag.decimate", Float.toString(cfg.quadDecimate)));
        cfg.quadSigma = Float.parseFloat(settings.getProperty("aprtag.blur", Float.toString(cfg.quadSigma)));
        return cfg;
    }

    private static AprilTagPoseEstimator.Config estimatorConfig(Properties settings) {
        return new AprilTagPoseEstimator.Config(
                Double.parseDouble(settings.getProperty("aprtag.size_meters", "0.1524")),
                Double.parseDouble(settings.getProperty("aprtag.fx", "850")),
                Double.parseDouble(settings.getProperty("aprtag.fy", "850")),
                Double.parseDouble(settings.getProperty("aprtag.cx", "300")),
                Double.parseDouble(settings.getProperty("aprtag.cy", "215"))
        );
    }

    private static void annotateTag(AprilTagDetection tag, Mat imageBuf) {
        for (int i = 0; i < 4; ++i) {
            Imgproc.line(imageBuf,
                    new Point(tag.getCornerX((i + 3) % 4), tag.getCornerY((i + 3) % 4)),
                    new Point(tag.getCornerX(i), tag.getCornerY(i)),
                    new Scalar(255, 0, 255), 2, 1);
        }

        Imgproc.putText(imageBuf, Integer.toString(tag.getId()),
                new Point(tag.getCenterX(), tag.getCenterY()), Imgproc.FONT_HERSHEY_PLAIN,
                2, new Scalar(255, 0, 255), 2);
    }

    private static void annotateCoords(double x, double y, double z, double yaw, Mat imageBuf) {
        String text = String.format("X=%.2f Y=%.2f Z=%.2f R=%.2f", x, y, z, yaw);
        Imgproc.putText(imageBuf, text,
                new Point(0, imageBuf.rows()), Imgproc.FONT_HERSHEY_PLAIN,
                2, new Scalar(255, 255, 0), 2);
    }

    private static void annotatePose(double x, double y, double yaw, int r, int g, int b, int thickness, int length, String label, Mat imageBuf) {
        int drawX = (int) ((x / FIELD_WIDTH) * imageBuf.cols());
        int drawY = imageBuf.rows() - (int) ((y / FIELD_HEIGHT) * imageBuf.rows());
        yaw *= -1;
        Imgproc.line(imageBuf, new Point(drawX, drawY),
                new Point(drawX + length * Math.cos(yaw), drawY + length * Math.sin(yaw)),
                new Scalar(r, g, b), thickness);
        Imgproc.rectangle(imageBuf,
                new Rect(new Point(drawX - 2, drawY - 2), new Point(drawX + 2, drawY + 2)),
                new Scalar(r, g, b), thickness + 1);

        if (label != null) {
            Imgproc.putText(imageBuf, label,
                    new Point(drawX, drawY), Imgproc.FONT_HERSHEY_PLAIN,
                    1.3, new Scalar(r, g, b), 1);
        }
    }

    static {
        try {
            FIELD = AprilTagFieldLayout.loadFromResource("/field.json");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() {
        this.detector.close();
    }
}
