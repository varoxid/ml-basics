
using SpamDetector;

MLModelSpam.ModelInput sampleData = new MLModelSpam.ModelInput()
{
    Content = @"Hi team, Just a reminder that we have our weekly project status meeting tomorrow at 10 AM in Conference Room B. Please bring your status reports and be prepared to discuss the timeline updates. Thanks, Sarah",
};


Console.WriteLine("Using model to make single prediction -- Comparing actual IsSpam with predicted IsSpam from sample data...\n\n");
Console.WriteLine($"Content: {@"Hi team, Just a reminder that we have our weekly project status meeting tomorrow at 10 AM in Conference Room B. Please bring your status reports and be prepared to discuss the timeline updates. Thanks, Sarah"}");
Console.WriteLine($"IsSpam: {@"false"}");


Console.ReadKey();

