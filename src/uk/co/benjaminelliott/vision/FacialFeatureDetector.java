package uk.co.benjaminelliott.vision;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FacialFeatureDetector {

	private CascadeClassifier faceClassifier;
	private CascadeClassifier eyeClassifier;
	private CascadeClassifier noseClassifier;
	private CascadeClassifier earClassifier;
	private CascadeClassifier mouthClassifier;
	
	private Scalar faceColor;
	private Scalar eyeColor;
	private Scalar noseColor;
	private Scalar earColor;
	private Scalar mouthColor;

	private VideoCapture cap;
	private Thread detectorThread;
	private volatile boolean running = false;

	private Mat frame;
	private Mat greyFrame;
	
	private ImageIcon image;
	private JFrame jFrame;


	public FacialFeatureDetector() {
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		
		faceClassifier = new CascadeClassifier("C:\\haar\\face.xml"); //TODO platform-specific
		eyeClassifier = new CascadeClassifier("C:\\haar\\eye.xml");
		noseClassifier = new CascadeClassifier("C:\\haar\\nose.xml");
		earClassifier = new CascadeClassifier("C:\\haar\\ear.xml");
		mouthClassifier = new CascadeClassifier("C:\\haar\\mouth.xml");
		
		faceColor = new Scalar(new double[]{0, 0, 0}); // black
		eyeColor = new Scalar(new double[]{255, 255, 255}); // white
		noseColor = new Scalar(new double[]{255, 0, 0}); // red
		earColor = new Scalar(new double[]{0, 255, 0}); // green -- note this is only one ear... TODO
		mouthColor = new Scalar(new double[]{0, 0, 255}); // blue
		
		frame = new Mat();
		greyFrame = new Mat();
		
		System.out.println("Everything instantiated.");
	}

	public static void main(String[] args) {
		FacialFeatureDetector fd = new FacialFeatureDetector();
		fd.startDetectingFromWebcam(0);
	}

	public void startDetectingFromWebcam(int deviceIndex) {
		cap = new VideoCapture(deviceIndex);

		if (!running) {
			running = true;
			
			detectorThread = new Thread() {
				@Override
				public void run() {
					while(running) {
						cap.read(frame);
						detectAllFeatures(frame);
					}
				}
			};

			detectorThread.start();
			
			System.out.println("Started detecting from webcam.");
		}
	}

	public void stopDetectingFromWebcam() {
		if (running) {
			running = false;
			if (cap != null)
				cap.release();
			System.out.println("Stopped detecting from webcam.");
		}
	}

	private void detectAllFeatures(Mat img) {
		Imgproc.cvtColor(img, greyFrame, Imgproc.COLOR_BGR2GRAY);

		List<Rect> faces = detectFeature(greyFrame, faceClassifier);
		
		if (!faces.isEmpty()) {
			Rect faceRect = faces.get(0);
			Mat faceOnly = greyFrame.submat(faceRect); //TODO only one?
			
			drawRectOnImg(faceRect, img, faceColor, 0, 0);

			List<Rect> eyes = detectFeature(faceOnly, eyeClassifier);
			if (eyes.size() == 2 && withinVerticalEyeThreshold(eyes.get(0), eyes.get(1))) {
				System.out.println("Eyes within vertical threshold.");
				
				
				for (Rect eye : eyes) {
					drawRectOnImg(eye, img, eyeColor, faceRect.x, faceRect.y);
				}
				
				List<Rect> noses = detectFeature(faceOnly, noseClassifier);
				for (Rect nose : noses) {
					drawRectOnImg(nose, img, noseColor, faceRect.x, faceRect.y);
				}
				
				List<Rect> ears = detectFeature(faceOnly, earClassifier);
				for (Rect ear : ears) {
					drawRectOnImg(ear, img, earColor, faceRect.x, faceRect.y);
				}
				
				int bottomOfEyes = eyes.get(0).y + eyes.get(0).height > eyes.get(1).y + eyes.get(1).height ? eyes.get(0).y + eyes.get(0).height : eyes.get(1).y + eyes.get(1).height;
				System.out.println("Bottom of eyes at "+bottomOfEyes);
				Mat belowEyes = faceOnly.submat(new Rect(0, 0 + bottomOfEyes, faceRect.width, faceRect.height - bottomOfEyes));
				
				List<Rect> mouths = detectFeature(belowEyes, mouthClassifier);
				for (Rect mouth : mouths) {
					drawRectOnImg(mouth, img, mouthColor, faceRect.x, faceRect.y + bottomOfEyes);
				}
			} else
				System.err.println("Eyes not within vertical threshold.");		
		}

		System.out.println("All features detected. Showing result...");

		showResult(frame);
		System.out.println("Result shown.");

	}
	
	private List<Rect> detectFeature(Mat greyImg, CascadeClassifier classifier) {
		MatOfRect results = new MatOfRect();
		classifier.detectMultiScale(greyImg, results);
		
		return results.toList();
	}
	
	private void drawRectOnImg(Rect toDraw, Mat image, Scalar color, int xOffset, int yOffset) {
		Point first = new Point(toDraw.x + xOffset, toDraw.y + yOffset);
		Point second = new Point(toDraw.x + toDraw.width + xOffset, toDraw.y + toDraw.height + yOffset);
		Imgproc.rectangle(image, first, second, color, 2);
	}
	
	public void showResult(Mat img) {
		MatOfByte matOfByte = new MatOfByte();
		Imgcodecs.imencode(".jpg", img, matOfByte);
		byte[] byteArray = matOfByte.toArray();
		try {
			InputStream in = new ByteArrayInputStream(byteArray);
			BufferedImage bufImage = ImageIO.read(in);
			if (image == null)
				image = new ImageIcon(bufImage);
			if (jFrame == null) {
				jFrame = new JFrame();
				jFrame.addWindowListener(new java.awt.event.WindowAdapter() {
			        public void windowClosing(WindowEvent winEvt) {
			            stopDetectingFromWebcam();
			            System.exit(0);
			        }
			    });
				jFrame.getContentPane().add(new JLabel(image));
				jFrame.pack();
				jFrame.setVisible(true);
			}
			else
				SwingUtilities.invokeLater(new Runnable() {

					@Override
					public void run() {
						image.setImage(bufImage);
						jFrame.invalidate();
						jFrame.repaint();
					}
					
				});
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private final int PIXEL_SCALE_FACTOR = 1; //TODO
	private final int EYE_VERTICAL_THRESHOLD = PIXEL_SCALE_FACTOR * 50;
	
	private boolean withinVerticalEyeThreshold(Rect eye0, Rect eye1) {
		return (Math.abs(eye0.y - eye1.y) <= EYE_VERTICAL_THRESHOLD && Math.abs(eye0.y + eye0.height - eye1.y - eye1.height) <= EYE_VERTICAL_THRESHOLD);
	}
}