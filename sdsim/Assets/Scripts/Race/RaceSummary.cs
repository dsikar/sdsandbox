﻿using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
public class RaceSummary : MonoBehaviour
{
    public Transform racerLayoutGroup;
    public GameObject racerSummaryPrefab;
    public GameObject racerSummaryFinalPrefab;

    int race_heat = 1;

    public void Init()
    {
        //clean out any previous summary...
        int count = racerLayoutGroup.childCount;
        for(int i = count - 1; i >= 0; i--)
        {
            Transform child = racerLayoutGroup.transform.GetChild(i);
            Destroy(child.gameObject);
        }

        // Now add one summary object per LapTimer
        LapTimer[] timers = GameObject.FindObjectsOfType<LapTimer>();

        // But first sort things according to place.
        Array.Sort(timers);
        List<string> summary_text = new List<string>();

        summary_text.Add("Heat,Place,Name,Best,Lap1,Lap2,Lap3,Total");

        for(int iT = 0; iT < timers.Length; iT++)
        {
            GameObject go = Instantiate(racerSummaryPrefab) as GameObject;

            RacerSummary s = go.GetComponent<RacerSummary>();

            s.Init(timers[iT], iT + 1, race_heat, summary_text);

            go.transform.SetParent(racerLayoutGroup);
        }

        try 
        {
            WriteHeatSummary(summary_text);
        }
        catch (Exception e) {
		 	Debug.LogError("Troubles writing heat summary: " + e);
		}

        race_heat += 1;
    }

    string GetLogPath()
    {
        if(GlobalState.log_path != "default")
            return GlobalState.log_path + "/";

        string path = Application.dataPath + "/../log/";

        // Make an attempt to create log if we can.
        try
        {
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);
        }
        catch
        {
            // Well. I tried.
        }

        return path;
    }

    void WriteHeatSummary(List<string> summary_text)
    {
        string filename = GetLogPath() + "HeatLog_" + race_heat.ToString() + ".csv";

	    StreamWriter writer = new StreamWriter(filename);

        foreach(String line in summary_text)
        {
            writer.WriteLine(line);
        }

        writer.Close();
    }

    public void Close()
    {
        this.gameObject.SetActive(false);
    }

    internal void InitFinal(List<Competitor> finalPlace)
    {
        //clean out any previous summary...
        int count = racerLayoutGroup.childCount;
        for (int i = count - 1; i >= 0; i--)
        {
            Transform child = racerLayoutGroup.transform.GetChild(i);
            Destroy(child.gameObject);
        }

        for (int iT = 0; iT < finalPlace.Count; iT++)
        {
            GameObject go = Instantiate(racerSummaryFinalPrefab) as GameObject;

            RacerSummary s = go.GetComponent<RacerSummary>();

            s.InitFinal(finalPlace[iT]);

            go.transform.SetParent(racerLayoutGroup);
        }
    }
}
