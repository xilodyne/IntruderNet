package xilodyne.utils.io.files;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.Random;

import org.apache.commons.io.FileUtils;

/*
 * Usage:  java -jar GetRandomFilesFromDir.jar 
 * 							<COPY or MOVE>
 * 							<Number of Files (if followed by %(no spaces), % of files in dir> 
 * 							<Source dir> 
 * 							<Dest Dir>
 */

public class GetRandomFilesFromDir {
	
	private static int numbOfFiles = 0;
	private static boolean doPercentage = false;
	private static boolean MOVE = false;
	private static String sSourceDir = "";
	private static String sDestDir = "";
	
	private static Random rand = new Random();
	private static HashSet<Integer> randUnique = new HashSet<Integer>();
	
	private static int TotFilesInDir = 0;
	
	private static File fSourceDir = null;
	private static File fDestDir = null;
	private static String[] fileNames = null;
	
	public static void main(String[] args) throws IOException {
		if (args.length < 4) {
			System.out.println("Usage: ");
			System.out.println("java -jar GetRandomFilesFormDir");
			System.out.println("  <COPY or MOVE>");
			System.out.println("  <Number of Files from Dir OR Pcnt of Files%>"); 
			System.out.println("  <source dir> <dest dir>");
		}
		if (args[0].compareTo("MOVE") == 0) {
			System.out.println("Do you wish to MOVE files?  Type MOVE to confirm:  ");
			BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
			String getMove = reader.readLine();
			if (getMove.compareTo("MOVE") == 0) {
				MOVE = true;
				System.out.println("MOVEed confirmed.");
			} else {
				System.out.println("MOVEed not confirmed");
				System.exit(1);
			}
			
		}

		if (args[1].endsWith("%")) {
			GetRandomFilesFromDir.doPercentage = true;
		}
		GetRandomFilesFromDir.numbOfFiles = Integer.parseInt(args[1]);
		GetRandomFilesFromDir.sSourceDir = args[2];
		GetRandomFilesFromDir.sDestDir = args[3];
		
		initFiles(sSourceDir, sDestDir);
		
		TotFilesInDir = getFileCount();
		
		System.out.println("Getting " + numbOfFiles + " files from a total of " + TotFilesInDir + " in " +sSourceDir);
		if (MOVE) {
			System.out.println("MOVING to : " + sDestDir);
		} else {
			System.out.println("Copying to: " + sDestDir);
		}
		System.out.println();
		
		for (int x = 1; x <= numbOfFiles; x++) {
			int getFile = getNextRand();
			String fileToMove = getFileAtNumber(getFile);
			copyFile(MOVE, fileToMove, sSourceDir, sDestDir);
			System.out.println("#" + x + " getting: " + getFile + " - " + fileToMove);
		}	
	}
	
	private static void initFiles(String srcDir, String desDir) {
		fSourceDir = new File(srcDir);
		fDestDir = new File (desDir);
	}
	
	private static int getFileCount() {
		fileNames = fSourceDir.list();
		return fSourceDir.listFiles().length;
	}
	
	private static int getNextRand() {
		int newRand = 0;
		boolean foundRand = false;
		while (!foundRand) {
			newRand = rand.nextInt(TotFilesInDir);
			
			if (!randUnique.contains((Integer)newRand)) {
				randUnique.add((Integer)newRand);
				foundRand = true;
			}
		}
		return newRand;
	}
	
	private static String getFileAtNumber(int getFile) {
		return fileNames[getFile];
	}
	
	private static void copyFile(boolean doMove, String fileToMove, String srcDir, String destDir) throws IOException {
		File srcFile = new File(srcDir + "/" + fileToMove);
		File destFile = new File(destDir + "/"+ fileToMove);
		
		if (doMove) {
			FileUtils.moveFile(srcFile, destFile);
		} else {
			FileUtils.copyFile(srcFile, destFile);
		}
	}
}
