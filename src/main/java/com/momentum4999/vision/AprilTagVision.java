package com.momentum4999.vision;

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import edu.wpi.first.apriltag.AprilTagFieldLayout;
import edu.wpi.first.apriltag.AprilTagPoseEstimator;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation3d;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.Closeable;
import java.io.IOException;
import java.util.Optional;
import java.util.Properties;

public class AprilTagVision implements Closeable {
    public static final AprilTagFieldLayout FIELD;

    private final Mat grayscaleBuffer = new Mat();
    private final AprilTagDetector detector = new AprilTagDetector();

    private final VisionNetworkTable table;
    private final AprilTagPoseEstimator poseEstimator;
    private final boolean annotate;

    public AprilTagVision(VisionNetworkTable table, Properties settings) {
        this.table = table;
        this.poseEstimator = new AprilTagPoseEstimator(estimatorConfig(settings));
        this.detector.setConfig(detectorConfig(settings));
        this.detector.addFamily(settings.getProperty("aprtag.family", "tag16h5"), 0);
        this.annotate = Boolean.parseBoolean(settings.getProperty("aprtag.annotate_video", "true"));
    }

    public void update(Mat colorBuffer) {
        if (!colorBuffer.empty()) {
            Imgproc.cvtColor(colorBuffer, grayscaleBuffer, Imgproc.COLOR_RGB2GRAY);

            int entries = 0;
            Translation3d pos = new Translation3d();
            double yaw = 0;
            for (AprilTagDetection tag : this.detector.detect(grayscaleBuffer)) {
                Transform3d detectedPose = this.poseEstimator.estimate(tag);
                Optional<Pose3d> possiblePose = FIELD.getTagPose(tag.getId());
                if (possiblePose.isPresent()) {
                    Pose3d pose = possiblePose.get();
                    Translation3d camPos = pose.getTranslation().minus(detectedPose.getTranslation());
                    double camYaw = pose.getRotation().getZ() - detectedPose.getRotation().getZ();
                    
                    pos = pos.plus(camPos);
                    yaw += camYaw;
                    entries++;
                }

                if (this.annotate) {
                    annotateTag(tag, colorBuffer);
                }
            }

            if (entries > 0) {
                pos = pos.div(entries);
                yaw = yaw / entries;

                if (this.annotate) {
                    annotatePos(pos.getX(), pos.getY(), pos.getZ(), yaw, colorBuffer);
                }
                this.table.updatePose(pos.getX(), pos.getY(), pos.getZ(), yaw);
            }
        }
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
                Double.parseDouble(settings.getProperty("aprtag.size_meters", "0.15")),
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

    private static void annotatePos(double x, double y, double z, double yaw, Mat imageBuf) {
        String text = String.format("X=%3.2f Y=%3.2f Z=%3.2f R=%3.2f", x, y, z, yaw);
        Imgproc.putText(imageBuf, text,
                new Point(0, imageBuf.rows()), Imgproc.FONT_HERSHEY_PLAIN,
                2, new Scalar(255, 255, 0), 2);
    }

    static {
        try {
            FIELD = AprilTagFieldLayout.loadFromResource("/field.json");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() throws IOException {
        this.detector.close();
    }
}
