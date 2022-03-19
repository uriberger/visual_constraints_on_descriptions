import java.nio.file.Path;
import java.nio.file.Paths;

import java.io.IOException;
import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.StringTokenizer;

import is2.data.SentenceData09;
import is2.lemmatizer.Lemmatizer;
import is2.transitionS2a.Parser;

public class LemmatizeAndParse {

	
	public static void main(String[] args) throws IOException {

		String modelsDirname = "models";
		String language = args[0];
		Path lemmatizerPath = null;
		Path parserPath = null;
		if (language.equals("English"))
		{
			lemmatizerPath = Paths.get(modelsDirname, "lemmatizer-eng-4M-v36.mdl");
			parserPath = Paths.get(modelsDirname, "per-eng-S2b-40.mdl");
		}
		else if (language.equals("German"))
		{
			lemmatizerPath = Paths.get(modelsDirname, "lemma-ger-3.6.model");
			parserPath = Paths.get(modelsDirname, "pet-ger-S2a-40-0.25-0.1-2-2-ht4-hm4-kk0");
		}
		else if (language.equals("French"))
		{
			lemmatizerPath = Paths.get(modelsDirname, "lemma-fra.mdl");
			parserPath = Paths.get(modelsDirname, "pet-fra-S2apply-40-0.25-0.1-2-2-ht4-hm4-kk0");
		}
		String dataset_name = args[1];
		Path captionFilePath = Paths.get("..", "..", "cached_dataset_files", dataset_name + "_" + language + "_dump_captions_train.txt");
		
		// create a lemmatizer and parser
		Lemmatizer lemmatizer = new Lemmatizer(lemmatizerPath.toString());
		Parser p = new is2.transitionS2a.Parser(parserPath.toString());
		
		FileWriter myWriter = new FileWriter("parsed.txt");
		File myObj = new File(captionFilePath.toString());
        Scanner myReader = new Scanner(myObj);
        while (myReader.hasNextLine()) {
			// Create a data container for a sentence
			SentenceData09 i = new SentenceData09();
			String data = myReader.nextLine();
			StringTokenizer st = new StringTokenizer(data);
			ArrayList<String> forms = new ArrayList<String>();
			forms.add("<root>");
			while(st.hasMoreTokens()) forms.add(st.nextToken());
			i.init(forms.toArray(new String[0]));

			// lemmatize a sentence; the result is stored in the stenenceData09 i 
			i= lemmatizer.apply(i);
		
			// apply the parser
			p.apply(i);
		
			// output the result
			for (int k=1;k< i.length();k++) myWriter.write(k+"\t"+i.forms[k]+"\t"+i.plemmas[k]+"\t_\t_\t"+i.ppos[k]+"\t_\t"+i.feats[k]+"\t_\t"+i.pheads[k]+"\t_\t"+i.plabels[k]+"\t_\t_\n");
			myWriter.write("\n");
        }
        myReader.close();
		myWriter.close();
	}

	
}
